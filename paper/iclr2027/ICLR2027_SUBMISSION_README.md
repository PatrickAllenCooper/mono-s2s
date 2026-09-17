# ICLR 2027 submission package — Monotonicity as an Architectural Bias for Robust Language Models

Prepared 2026-09-16 from the updated Overleaf content (`paper/overleaf_update_20260916/`; its `CHANGELOG.md` lists every content change relative to the original Overleaf draft). This folder is a complete, self-contained ICLR project: upload it as-is.

## Files

| File | Role |
|---|---|
| `monotone_llms_iclr2027.tex` | Main file. Anonymous; `\iclrfinalcopy` is commented out (submission mode with the line ruler). |
| `example_paper.bib` | Bibliography (identical to the Overleaf update: 61 verified entries). |
| `order_preservation_depth.pdf` | Figure 1 (regenerated from `curc_validation_results/` at its printed size, 3.45 in wide, so all labels are ≥ 6.3 pt in print). |
| `iclr2027_conference.sty`, `iclr2027_conference.bst`, `natbib.sty`, `fancyhdr.sty`, `math_commands.tex` | Official ICLR 2027 style files, byte-identical to `iclr-2027-style-files.zip` (fetched 2026-09-16). |
| `monotone_llms_iclr2027.pdf` | Compiled with Times (T1) metrics. 26 pages: main text pp. 1–9 (Section 7 ends about 6 lines above the bottom of p. 9), AI use statement pp. 9–10, ethics + reproducibility statements p. 10, references pp. 10–14, appendix pp. 15–26. |
| `make_figure.py` | Regenerates Figure 1 from the result JSONs (not needed for submission). |

## Compliance checklist (ICLR 2027 author guidelines, as of 2026-09-16)

- [x] **Main text ≤ 9 pages** — title through the last line of Section 7 ends on page 9 with about 6 lines of slack; the "AI use statement" heading and its first lines fill the rest of page 9 (statements do not count toward the limit). The limit is strictly enforced, so after your first Overleaf compile confirm that the last sentence of Section 7 ("…architecture-dependent cost in clean performance.") is still on page 9. If pdfLaTeX ever breaks lines differently, cheap slack: `\includegraphics[width=0.60\linewidth]` for Figure 1, or shorten the Table 1/2 captions.
- [x] **AI use statement (required)** — `\subsection*{AI use statement}` after the main text, before references, following the template's structure (required-disclosure tasks, recommended tasks, review, responsibility). **Authors: verify it matches your actual usage** — it discloses AI assistance in implementing the pipeline, computing the reported statistics and drafting their interpretation, checking and correcting the Section 5 proofs, editing text, finding/verifying references, and producing Figure 1. The same disclosure must also be entered in the OpenReview submission form.
- [x] **Ethics statement** and **Reproducibility statement** (recommended) — present, before references. The reproducibility statement currently says code and per-seed result files "will be released with the camera-ready version"; if you attach an anonymized code/results link as supplementary material instead, change that sentence (any link must be fully anonymous, with no visitor tracking).
- [x] **Double-blind** — no names, e-mails, affiliations, acknowledgments, funding, or repository names in the tex, bib, or PDF. Author block reads "Anonymous authors / Paper under double-blind review". Self-citations: none.
- [x] **Style** — `\documentclass{article}` + `\usepackage{iclr2027_conference,times}`; official style files unmodified; `\bibliographystyle{iclr2027_conference}`. Added packages: `[T1]{fontenc}` (makes Times metrics identical under XeTeX; harmless under pdfLaTeX), `enumitem` (compact contributions list), `hyperref` (as in the official template), and three float-spacing lengths (`\textfloatsep`, `\abovecaptionskip`, `\belowcaptionskip`) — none change font, margins, or text size.
- [x] **Appendix after references** in the same PDF (allowed). Appendix A.1–A.12 contains everything removed from the main text (extended related work, the pre-normalization remark and the proof of Proposition 3.3, the full mechanism derivations including the persistence lemma with its proof, all protocols, XSUM and training-dynamics tables, the screening table, ablation, trigger transfer, and the extended discussion).
- [x] Compiles with no undefined references or citations, no overfull boxes, and no BibTeX warnings.
- [x] Independently verified (2026-09-16, 81-agent review/refutation pass over compliance, content preservation vs. the Overleaf version, mathematics, numbers vs. `numbers.json`, LaTeX integrity, and clarity): all confirmed findings applied (see `../overleaf_update_20260916/CHANGELOG.md` §6 for the three corrections that also apply to the Overleaf version).

## What was compressed (main text) and where it went

| Main-text section | Change | Moved to |
|---|---|---|
| Introduction | ~1,500 → ~750 words; examples kept, contributions tightened | — |
| Related work | 3 paragraphs → 2 | A.1 Extended Related Work |
| §3 Monotone Transformers | Definitions merged; proof of Prop. 3.3, Remark on pre-LN blocks, and the extended initialization discussion moved | A.2 |
| §4 Setup | Attack, control, and probe protocols summarized | A.5–A.8 (full text) |
| §5 Mechanism | Lemmas 5.1–5.3 and Assumption 5.2 retained verbatim; persistence lemma summarized in prose | A.4 (full derivations, sign-coherence discussion, scope; Lemma A.2 persistence with proof), A.3 (proofs) |
| §6 Results | Training-dynamics table, XSUM table, length/brevity paragraph, screening table moved; ROUGE and HotFlip tables side by side; one-seed HotFlip check summarized | A.9 Additional Results; A.10 Ablation; A.11 Trigger transfer |
| §7 | Discussion + Conclusion merged into one section | A.12 Extended Discussion (full original text) |

All numbers, caveats, and claims of the Overleaf version are retained (either in the main text or in the appendix); wording in the main text was tightened for length only.

## Submitting

1. Upload the files in this folder to a fresh Overleaf project and compile with pdfLaTeX; expect 26 pages, no undefined references.
2. Confirm that Section 7 ends on page 9 (see checklist).
3. OpenReview: abstract deadline **Sep 18, 2026, 11:59 PM AoE**; full paper **Sep 25, 2026, 11:59 PM AoE**. Authors cannot be changed after the abstract deadline. Fill in the AI-use disclosure in the submission form as well (the guidelines require it in both places).
4. Before the full-paper deadline, decide on the open items in `../overleaf_update_20260916/CHANGELOG.md` §1 (ablation robustness numbers; multi-seed HotFlip; the UAT-truncation and Pythia loss-masking re-runs, which would make the caveats in §4 unnecessary).
