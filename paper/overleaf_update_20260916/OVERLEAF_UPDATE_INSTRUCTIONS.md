# Exactly what to update on Overleaf

Project: the one exported as `Monotone_LLMs.zip` (main file `example_paper.tex`, TMLR style).

## A. Files (3 uploads, ~2 minutes)

1. **`example_paper.tex`** → Overleaf › `example_paper.tex`: open the existing file, select all, paste the contents of the bundled `example_paper.tex` (or drag-and-drop the file onto the file tree and confirm "Replace").
2. **`example_paper.bib`** → replace the existing `example_paper.bib` the same way.
3. **`order_preservation_depth.pdf`** → upload to the project root (same folder as `example_paper.tex`). It is referenced by `\includegraphics[width=\linewidth]{order_preservation_depth.pdf}` (new Figure 1).
4. Recompile. Expected: 27 pages, no undefined references or citations (verified locally with tectonic against the bundled `tmlr.sty`/`icml2025.bst`; nothing else in the project needs to change).

Do **not** replace `main.tex`, `supplementary_results.tex`, `archive/`, or the style files.

## B. Review pass (30–60 minutes)

Search the new `example_paper.tex` for `%% [CHANGE` — 86 markers, each naming its category and, where it replaces text, quoting what was there. Go through them in this order:

1. `method-as-implemented` (Sections 3–4, 6.1) — the A = I / `|W_pre|`-initialisation correction. Read the new "Implementation Scope" paragraph and the two paragraphs in Section 3 that now end with "…all models trained in this paper use this instantiation".
2. `math-rigor` (Section 5 and Appendix A.1) — Assumption 5.2, Lemmas 5.3/5.4 and their proofs, Remark 3.8, the universal-approximation sentence, the GELU caveat. `math-rigor, minor` markers are optional precision edits.
3. `factual` — MLP-both (not MLP-in) is the T5 analogue; SAMSum; hardware; "broader constraint coverage" wording; epochs.
4. `rigor` / `new result` / `new discussion` — absolute-nats caveat (Table 6, Table 7), order-preservation section + figure, controls paragraph, discussion paragraph, Appendix C (ablation) and D (trigger transfer).
5. `citation` / `new citations` — replaced/removed references and the three new Related-Work sentences.

Then read `CHANGELOG.md` §1 (eleven decisions) and act on 4, 5, 7, 8 and 9 in particular.

## C. Before submission (not done here)

- Fill in the ablation robustness numbers (Appendix C) and multi-seed HotFlip (Table 4) if the CURC jobs finish in time; otherwise the text already states the current status honestly.
- Choose the venue template. The content is format-independent; for ICML/ICLR the main text must be cut to 8/9 pages (move Section 5's proofs, the controls table, or the order-preservation figure to the appendix).
- Delete the `%% [CHANGE …]` comments once reviewed (optional; they do not render).
- Anonymise the `\author` block if submitting double-blind.
