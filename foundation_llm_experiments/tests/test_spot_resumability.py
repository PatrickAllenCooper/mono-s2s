"""
Tests for spot-deallocation hardening primitives.

These exercise the durability and resume invariants we rely on to make
Stages 4-6 survive unplanned termination on Azure spot VMs:

- atomic_save_json never leaves a half-written file observable.
- append_jsonl + load_jsonl tolerates a truncated final record.
- compute_perplexity_resumable skips already-scored batches and produces
  the same answer whether it runs once or is interrupted and resumed.
"""

import json
import os

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from utils.common_utils import (
    atomic_save_json,
    load_json,
    load_json_safe,
    append_jsonl,
    load_jsonl,
    compute_perplexity_resumable,
)


class TestAtomicSaveJson:
    def test_roundtrip(self, tmp_path):
        path = tmp_path / "out.json"
        atomic_save_json({"a": 1, "b": [2, 3]}, str(path))
        assert load_json(str(path)) == {"a": 1, "b": [2, 3]}

    def test_creates_parent_directory(self, tmp_path):
        path = tmp_path / "nested" / "deep" / "out.json"
        atomic_save_json({"x": 1}, str(path))
        assert path.exists()

    def test_no_tmp_leftover_on_success(self, tmp_path):
        path = tmp_path / "out.json"
        atomic_save_json({"x": 1}, str(path))
        assert not (tmp_path / "out.json.tmp").exists()

    def test_overwrite_preserves_previous_on_fresh_read(self, tmp_path):
        path = tmp_path / "out.json"
        atomic_save_json({"v": 1}, str(path))
        atomic_save_json({"v": 2}, str(path))
        assert load_json(str(path)) == {"v": 2}


class TestLoadJsonSafe:
    def test_missing_returns_default(self, tmp_path):
        assert load_json_safe(str(tmp_path / "missing.json"), default={"k": 0}) == {"k": 0}

    def test_corrupt_returns_default(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("{not valid json")
        assert load_json_safe(str(path), default=None) is None


class TestJsonl:
    def test_append_and_load(self, tmp_path):
        path = tmp_path / "records.jsonl"
        for i in range(5):
            append_jsonl(str(path), {"idx": i, "value": i * i})
        assert load_jsonl(str(path)) == [
            {"idx": i, "value": i * i} for i in range(5)
        ]

    def test_tolerates_truncated_trailing_line(self, tmp_path):
        """A crash mid-write leaves a malformed final line; prior records stay."""
        path = tmp_path / "records.jsonl"
        append_jsonl(str(path), {"idx": 0})
        append_jsonl(str(path), {"idx": 1})
        with open(path, "a") as f:
            f.write('{"idx": 2, "incom')
        records = load_jsonl(str(path))
        assert records == [{"idx": 0}, {"idx": 1}]

    def test_missing_file_returns_empty(self, tmp_path):
        assert load_jsonl(str(tmp_path / "nope.jsonl")) == []


class _FixedLossModel(nn.Module):
    """Deterministic stand-in for a language model that returns a known loss."""

    def __init__(self, loss_value):
        super().__init__()
        self.loss_value = loss_value
        self._param = nn.Parameter(torch.zeros(1))

    def forward(self, input_ids=None, labels=None, **kwargs):
        class _Out:
            pass
        out = _Out()
        out.loss = self._param.sum() + self.loss_value
        return out

    def eval(self):
        return self


def _make_dataloader(num_batches, batch_size=2, seq_len=4):
    input_ids = torch.ones(num_batches * batch_size, seq_len, dtype=torch.long)
    attn = torch.ones_like(input_ids)
    ds = TensorDataset(input_ids, attn)

    def _coll(batch):
        ids = torch.stack([b[0] for b in batch])
        am = torch.stack([b[1] for b in batch])
        return {"input_ids": ids, "attention_mask": am}

    return DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=_coll)


class TestComputePerplexityResumable:
    def test_matches_full_run_when_no_checkpoint(self, tmp_path):
        model = _FixedLossModel(loss_value=1.5)
        loader = _make_dataloader(num_batches=4)
        out = compute_perplexity_resumable(
            model, loader, torch.device("cpu"),
            progress_path=str(tmp_path / "p.json"),
        )
        assert out["total_tokens"] == 4 * 2 * 4
        assert out["loss"] == pytest.approx(1.5, rel=1e-6)
        assert out["perplexity"] == pytest.approx(float(np.exp(1.5)), rel=1e-6)

    def test_resume_from_mid_run_gives_same_result(self, tmp_path):
        """Simulate a deallocation after 2 of 4 batches: final answer must be identical."""
        progress_path = str(tmp_path / "p.json")

        model = _FixedLossModel(loss_value=1.5)
        state = {
            "batches_done": 2,
            "total_loss_weighted": 1.5 * (2 * 2 * 4),
            "total_tokens": 2 * 2 * 4,
        }
        with open(progress_path, "w") as f:
            json.dump(state, f)

        loader = _make_dataloader(num_batches=4)
        resumed = compute_perplexity_resumable(
            model, loader, torch.device("cpu"),
            progress_path=progress_path,
        )

        fresh = compute_perplexity_resumable(
            model, _make_dataloader(num_batches=4), torch.device("cpu"),
            progress_path=str(tmp_path / "fresh.json"),
        )
        assert resumed["total_tokens"] == fresh["total_tokens"]
        assert resumed["loss"] == pytest.approx(fresh["loss"], rel=1e-6)

    def test_progress_file_persisted(self, tmp_path):
        progress_path = str(tmp_path / "p.json")
        model = _FixedLossModel(loss_value=2.0)
        loader = _make_dataloader(num_batches=3)
        compute_perplexity_resumable(
            model, loader, torch.device("cpu"),
            progress_path=progress_path, flush_every=1,
        )
        saved = load_json(progress_path)
        assert saved["batches_done"] == 3
        assert saved["total_tokens"] == 3 * 2 * 4


class TestResumeSkipIndex:
    """End-to-end pattern used by stage 6: JSONL append with idx-based skip."""

    def test_round_two_does_not_reprocess_completed(self, tmp_path):
        path = tmp_path / "hotflip.jsonl"

        def _run(start_from_existing, max_examples, processed_log):
            existing = load_jsonl(str(path))
            done = {r["idx"] for r in existing}
            for i in range(max_examples):
                if i in done:
                    continue
                processed_log.append(i)
                append_jsonl(str(path), {"idx": i, "value": i})

        first_log = []
        _run(False, 5, first_log)
        # Simulate a crash by truncating the last record mid-write
        with open(path, "a") as f:
            f.write('{"idx": 5, "value": 5')

        second_log = []
        _run(True, 8, second_log)

        assert first_log == [0, 1, 2, 3, 4]
        assert set(second_log) == {5, 6, 7}
        records = load_jsonl(str(path))
        assert sorted(r["idx"] for r in records) == list(range(8))
