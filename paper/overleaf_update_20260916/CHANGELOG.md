# Monotone LLMs — Overleaf update, 2026-09-16

Base: the Overleaf export `Monotone_LLMs.zip` (`example_paper.tex`, 1164 lines, TMLR style).
Deliverables in this bundle:

| File | What to do with it |
|---|---|
| `example_paper.tex` | Replace the Overleaf file of the same name. Every deliberate change is marked by a `%% [CHANGE: <category>] …` comment on its own line or at the end of the changed line (86 markers; grep for `%% [CHANGE`); the comments are safe to leave in or delete. |
| `example_paper.bib` | Replace the Overleaf file of the same name (regenerated: 16 entries removed, 13 corrected, 24 added; see §4). |
| `order_preservation_depth.pdf` | Upload to the Overleaf project root (new Figure 1). |
| `example_paper.pdf` | Compiled result (tectonic, 27 pages in TMLR single-column format) for reference. |
| `original_vs_updated.diff` | Unified diff against the Overleaf version. |
| `numbers.json` | Every statistic used in the new tables/prose, computed from `curc_validation_results/`. |

Change categories used in the `%% [CHANGE …]` markers:
`results` (new numbers), `new result` (new experiment), `method-as-implemented`, `factual`, `protocol`, `metric`, `math-rigor`, `citation`, `new citations`, `rigor`, `clarity`, `grammar`, `cleanup`, `new discussion`, `new protocol`.

---

## 1. Decisions that need an author (please read first)

1. **Method description now matches the code (Sections 3, 4, 6.1).** The Overleaf text said the experiments use a `p = 64` semantic projection `F(h) = h + A†g(Ah)` with `g` initialised near zero. Nothing of the kind exists in the trained models: `hpc_version/utils/common_utils.py` constrains the full-width FFN matrices `wi`/`wo` (i.e. the paper's own `A = I` case) and initialises `V = softplus⁻¹(|W_pre| + 1e-4)`. Three independent reviewers plus nine adversarial re-checks confirmed this. The general `A`-framework is retained word for word as a framework; the paper now says `A = I` was used in all experiments and that the 64 probe directions are a post-hoc diagnostic. **This is the one change that touches "theory and framing"; it is not optional if the paper is to be truthful, but you should read the new wording.**
2. **Section 5 theory corrected (see §3).** The saturation assumption was stated for `z → +∞`, which is false for ReLU (T5) and GELU (Pythia); the persistence lemma was false for ReLU and vacuous for smooth activations; the attenuation lemma had no proof. All lemmas keep their names and roles, but their statements/proofs changed. Confirmed by 3-of-3 adversarial refuters in every case.
3. **Absolute-vs-relative degradation on Pythia (§6.3, Table 6, Table 7, Discussion).** Monotone Pythia's clean NLL is ~1.9× the baseline's, so relative degradations flatter it. In absolute nats the UAT effect is *not* smaller for the monotone model (0.40 ± 0.14 vs 0.32 ± 0.01), and the HotFlip/random/query gaps shrink to 0.77 vs 0.82, 0.66 vs 0.91, 1.17 vs 1.39 nats. The draft now reports both and hedges the UAT claim. You may prefer to frame this differently, but a reviewer will compute it.
4. **Order-preservation diagnostic is a negative/neutral result (new §6.3, Figure 1).** The monotone T5 encoder is *less* order-preserving along learned semantic directions at intermediate depth (0.42–0.45 vs ~0.9 for the baseline at layers 3–5, all five seeds) and comparable at the output. It is written up honestly as "constraint ≠ end-to-end semantic order preservation". Decide whether to keep it in the main text (my recommendation), move it to the appendix, or drop it.
5. **Attribution ablation (Appendix C).** Training dynamics for `sign_frozen`/`abs_init_free` show the clean-quality cost comes from the `|W_pre|` initialisation, not the constraint. **Their robustness evaluation is the single most important open experiment**: if `abs_init_free` is also robust, the causal story changes. The appendix states that this evaluation is left to future work; replace that sentence with the numbers if they land before the deadline (jobs are queued on CURC).
6. **SAMSum is gone.** Every CURC run loaded 0 SAMSum examples (`data_statistics.json: "samsum": 0`, HF script-dataset fallback). No SAMSum numbers were ever in the tables; the appendix claim is removed. A fix for future runs is queued as a separate task.
7. **HotFlip on T5 uses one training seed (2024, n = 200)** plus the earlier seed-42/n = 100 run for corroboration; the other seeds' HotFlip jobs were still running. Table 4 says so. Replace with the multi-seed mean ± s.d. if those land.
8. **Venue/format.** The Overleaf file is TMLR-formatted; you said ICML in the latest request and ICLR 2027 earlier (abstract 18 Sep, paper 25 Sep). The content update is format-independent; say which template you want and I will port it (main-text page budget will require moving material to the appendix for ICML/ICLR).
9. **Two methodological problems found by the code audit — both written into the paper as caveats, both fixable by a cheap re-run (task chips queued).**
   * *T5 universal triggers never reach the model on long articles.* `stage_5_uat_attacks.py` prepends the trigger and then **left-truncates** the encoder input to 512 tokens, so for any article longer than ~505 tokens the trigger is discarded and the "attacked" input equals the clean input. This is almost certainly why UAT effects on T5 are <1 %. The paper now says so (Section 4.1, Section 6.2) and reads the UAT result as a lower bound. Re-running Stage 5/5b for the four seeds after the fix takes a few GPU-hours per seed.
   * *Pythia attack losses score the inserted tokens themselves.* All Pythia attack metrics use `labels = input_ids` of the perturbed sequence, so the 10 trigger/substituted tokens are prediction targets and part of every reported NLL increase (including the 65 % random-substitution effect) is just their low likelihood. Comparisons between the two models remain valid (identical scoring), but absolute figures are inflated; the paper says so in Section 4.2. A masked-label variant should be run for the three seeds.
10. **Table 6 baseline column.** The baseline column is identical in every seed row (PPL 6.72, SR 69.0 %), whereas the re-run inside the transfer pipeline (Table 7) gives 68.5–69.0 %. Confirm whether one baseline checkpoint was reused across seeds in the original stage-4/6 evaluation and say so in the caption.
11. **Optional precision edits.** A few `%% [CHANGE: math-rigor, minor]` edits (preorder remark, differentiability qualifier in Lemma 5.1, the explicit chain-rule decomposition, the Hirsch wording, hedges in the Discussion) address points that the adversarial reviewers judged *acceptable as originally written*. They are strictly more precise, but if you want to minimise edits to the theory prose they can be reverted individually.

## 2. Results updated (all numbers from `numbers.json`)

* **Abstract**: attack success 60 % (matched fine-tuned baseline; 69 % for pretrained) → 17 %; ~5 % relative ROUGE-L cost over five seeds; controls sentence added; stray `..` fixed.
* **Table 1 (training dynamics)**: five-seed mean ± s.d., train and validation loss at epochs 1 and 7; the old table mixed train/val and said the baseline ran 6 epochs (both run 7).
* **Tables 2–3 (ROUGE)**: full CNN/DailyMail (11,490) and XSUM (11,333) test sets, five seeds, ROUGE F1 × 100 mean ± s.d.; old tables were seed 42, n = 200. Prose updated: CNN/DM gap 4.9 % ROUGE-L (was 3.2 %), per-example paired t-tests p < 1e-57 with small effect sizes (d 0.15–0.24); XSUM gap 2.1 % (d ≤ 0.14). The old "confidence intervals overlap substantially" sentence is no longer true and was replaced.
* **Length/brevity paragraph**: numbers recomputed (75.4/74.9/62.3 tokens; brevity penalty 0.955 vs 0.962). The old paragraph was internally inconsistent (called 11.3 "higher variance" than 12.0).
* **Table 4 (HotFlip)**: seed 2024, n = 200: 19.9 %/68.5 %/+0.40, 15.1 %/59.5 %/+0.35, 4.9 %/17.0 %/+0.14; relative reductions 71 %/68 %; t = 9.04 (p = 6.9e-18), t = 10.49 (p = 6.5e-23); Cohen's d = 0.91; absolute-nats sentence added; "paired t-tests" corrected to independent-samples (the code uses `ttest_ind`).
* **UAT (T5)**: four seeds, 0.33 ± 0.05 % / 0.56 ± 0.06 % / 0.74 ± 0.09 % NLL increase; protocol corrected to 5 restarts/100 iterations/1,000 + 1,500 examples (the old text described the Pythia setting); candidate pool and search described as implemented; transfer statement added (100-example transfer subset); **left-truncation caveat added (see decision 9)**.
* **New §6.3 Order Preservation Along Semantic Directions** + Figure 1 (`order_preservation_depth.pdf`).
* **Pythia (§6.4)**: Table 6 gains an absolute-ΔNLL column and the 15 % threshold; new paragraph + Table 7 "Transfer and gradient-free controls" (three seeds: HotFlip crafted-on × evaluated-on, random substitutions, gradient-free query attack, with absolute nats); "MLP-in is the direct analogue of T5" corrected to MLP-both throughout (Setup, Results, Discussion, Conclusion, Contribution 4) — the T5 code constrains both projections.
* **Discussion**: one new paragraph on what the controls and the order-preservation result imply; three hedges (`math-rigor, minor`).
* **Appendix**: SAMSum removed; training-set size corrected (242,774, not ~150K); linear-decay schedule and best-checkpoint selection stated; seed protocol stated precisely per metric; hardware sentence updated; Pythia protocol completed (optimizer, LR/warmup differences between baseline and monotone recovery runs, batch/sequence length, screening budgets per scope, precision per stage); new Appendix "Attribution Ablation" (Table 8) and "Universal Trigger Transfer" (Table 9); `\nocite{langley00}` removed.

## 2b. Clarity and flow pass (last step)

Applied after the independent clarity review: restricted "modest cost" claims to the encoder–decoder setting (intro, conclusion); removed "jailbreak-style" from the contributions/discussion (no jailbreak experiment is run); softened "isolating monotonicity itself as a causal factor" and "structural constraints alone" (ablation robustness pending); removed three duplicated passages (metric definitions, the 1.94-point recovery description, the 71 %/67 % reductions) and the untested warmup claim; reconciled the initialization-cost attribution with the "trade-off" sentence; fixed the leftover "requires constraining a larger fraction of each block"; put the T5 and Pythia clean-performance costs in like units (1.35× vs 5.61× perplexity); added a Section 3 roadmap, an intuition paragraph before Assumption 5.2, bold lead-ins (Attacks / Loss convention / Controls) in Section 4.2, paragraph breaks in the controls and discussion paragraphs, and a results roadmap that mentions the new Section 6.3; expanded MLP/NLL/pp on first use; renamed the appendix "Experimental Setup" to "Seeds, Statistical Tests, and Compute"; made the abstract's ROUGE-L cost "2–5 % on two benchmarks" and "input projection alone".

## 3. Mathematics (Sections 3 and 5, Appendix A) — what changed and why

All confirmed by three independent adversarial reviewers each unless marked *minor/optional*.

| Item | Original | Problem | Now |
|---|---|---|---|
| Assumption 5.2 | `σ'(z) → 0 as z → +∞` | False for ReLU/GELU/softplus (σ' → 1); they saturate as z → −∞ | `|σ'(z)| → 0 as z → −∞`, `|σ'|` non-decreasing on a tail `(−∞, z*]`; ReLU z* = 0, GELU z* = −√2 |
| Lemma 5.3 (attenuation) | "contribution vanishes"; proof pointer went to the proof of Lemma 5.1 | No proof; "contribution of a unit" undefined; needs bounded downstream gradient (counterexample otherwise) | Explicit rank-one decomposition of `J_g`, quantitative bound `≤ ε G Σ‖W₂[:,j]‖‖W₁[j,:]‖`, new proof A.1.3 |
| Text after Lemma 5.3 | "attenuates gradient magnitude" | `J_T ≥ I`; saturating a unit can *increase* the norm for mixed-sign downstream gradients | States what is shown (FFN term suppressed), attenuation only under sign coherence; notes nonnegativity is not needed for the bound |
| Lemma 5.4 (persistence) | saturated (`σ' = 0`) at s′ ⇒ saturated at s′ + δ, δ ≥ 0 | False for ReLU (δ ≥ 0 can re-activate a dead unit); vacuous for smooth σ (σ' never exactly 0) | ε-saturation, persistence under δ ≤ 0; proof A.1.4 rewritten |
| §5.4 narrative | constraint "drives units into saturated regimes from which they do not recover under monotone increases" | Not established; direction reversed for the real activations | Conditional statement + explicit **Scope** paragraph (residual/attention paths unconstrained; controls needed) |
| §5 opening | "mechanistic explanation" | Lemmas are conditional | "candidate mechanism" |
| Prop. 3.6 remark | nonnegative-weight nets with any non-decreasing σ are universal approximators (Daniëls & Velikova) | Theorem is for sigmoidal σ; nonneg-weight ReLU nets are convex ⇒ not universal | Corrected; ReLU family noted as restricted |
| Pythia/GELU | Pythia models called monotone | GELU' < 0 for z < −0.75 (min ≈ −0.13) ⇒ Prop. 3.6 does not apply exactly | Caveat added in §4.2 and in the reparameterisation paragraph |
| Pre-LN block | `F(h) = h + g(h)` | T5/GPT-NeoX apply LayerNorm before the branch: `F(h) = h + g(LN(h))`, LN unconstrained | New Remark 3.8: the constrained/analysed object is the branch `g` |
| Init | "initialise g near the zero map, V ≪ 0, biases zero" | Code: `V = softplus⁻¹(|W_pre| + 1e-4)`; T5 has no FFN biases | Corrected (also explains the 4.99 initial loss) |
| Lemma 5.1 *(minor)* | "for all s" | Undefined at ReLU kinks; proved for 2 layers only | Differentiability qualifier + Clarke-Jacobian sentence; general-depth line in A.1.2 |
| Chain rule *(minor)* | objective "on semantic coordinates" | loss is a function of h′ ∈ ℝ^d, not s′ | Explicit `J_F = I + A†J_gA` decomposition; residual term unaffected |
| Preorder *(minor)* | — | ⪯ is a preorder, partial order iff A injective | One-sentence remark |
| Hirsch sentence *(minor)* | "converge almost surely; derivatives tend to zero" | genericity, not probability; unrelated to sensitivities | Reworded |
| Second-round fixes (independent re-review of the revised text) | — | logistic activation saturates on both tails (was listed as one-sided); L-layer Jacobian display omitted intermediate weights; Clarke-Jacobian sentence needed the convex-hull argument and local Lipschitz hypothesis; Lemma 5.3's "in particular" clause needed Assumption 5.2 as a stated hypothesis; sign-coherence remark needs σ' ≥ 0 (T5 only); T(s) ≠ the pre-LN sublayer map (J_LN factor made explicit); ⪯ vs ≤ on ℝ^p; GELU' zero at −0.752 (not −0.75), min at −√2; GPT-NeoX uses LayerNorm not RMSNorm; `Section~1` hard-coded → `\ref{sec:intro}` | All applied |

## 4. Bibliography

**Removed (16):** ICML-template placeholders `langley00, Samuel59, mitchell80, kearns89, Newell81, DudaHart2nd, MachineLearningI, anonymous`; `PromptInjectionWikipedia` (uncited Wikipedia); `Carlini2023AreAlignedLLMsRobust` (fabricated title/authors, uncited); `Gowal2018Certified` (conflates two papers, uncited); `Wei2023JailbreakSurvey` (title/author/arXiv from three different papers, uncited); `howe2024scaling` (duplicate); **`Weber2021CertifyingMonotonicNetworks` (could not be found anywhere — fabricated authors and arXiv id; it was cited in Related Work and is replaced by Liu et al. 2020 + Sivaraman et al. 2020 with the sentence reworded to what they show)**; `salah2026jailbreaking` (unrefereed ResearchGate PDF, was the lead citation of the intro; replaced by Wei, Haghtalab & Steinhardt, NeurIPS 2023); `wang2025understanding` (a survey of *transfer learning* cited for *attack transferability*; sentence re-cited to Zou et al. 2023).

**Corrected metadata (13):** `Gupta2016MonotonicLattices` (JMLR 17(109), 8 authors; was "NeurIPS 2016", 4 authors), `Robey2023SmoothLLM` (authors are Wong/Hassani/Pappas, not Wang/Kolter), `deng2024multilingual` (ICLR 2024; stray ". 2023" in title), `feyzmahdavian2017stability` (2018, DOI), `Wang2021AdvGLUE` (had the *GLUE* authors), `Zhu2023AutoDAN` (had a fabricated author list and the *other* AutoDAN's title), `Zou2023…` (Fredrikson missing), `howe2024effects` (no venue; now ICML 2025/PMLR 267), `jukna2012boolean` ("and others" artifact), `Daniels2010Monotone` (Hennie, not "Hani"), `You2017DeepLattice` (three fabricated first names), `sayeedi2025jailbreaktracer` (author/volume/DOI; and the sentence citing it was wrong — it is a prompt classifier, not output-distribution detection), `sartor2025advancing` (ICML 2025).

**Added (24), each opened on a primary listing page (arXiv/ACL Anthology/PMLR/NeurIPS/Crossref):** `athalye2018obfuscated, carlini2019evaluating, tramer2020adaptive` (gradient-masking methodology for the controls), `wei2023jailbroken, jain2023baseline, zou2024circuit, jia2019certified` (defenses), `shin2020autoprompt, guo2021gradient, jin2020bert, cheng2020seq2sick` (token-level attack lineage incl. seq2seq), `liu2020certified, sivaraman2020counterexample, wehenkel2019unconstrained, nolte2023expressive, kitouni2023robust, igel2024smooth, mikulincer2022size, chorowski2015learning, incer2018adversarially` (monotone/nonnegative networks, incl. the monotonicity-for-robustness precedent), `hirsch2006monotone, angeli2003monotone, winston2020monotone` (monotone systems/operators), `kim2021lipschitz` (Lipschitz Transformers).

Kept although uncited (real, harmless): `Carlini2017DefensiveDistillation, Madry2018Towards, Zhang2019Tradeoff`.

## 5. Things I did not do

* Did not change the TMLR template, author block, or bibliography style (`icml2025.bst` with `tmlr.sty` compiles fine).
* Did not touch the intro's motivation, examples, or contributions beyond the marked factual edits.
* Did not add the T5-base / Pythia-2.8B / 6.9B scale-up (not finished; T5-base monotone re-running after the bf16 fix).

---

## 6. Addendum (2026-09-16, evening) — corrections found while preparing the ICLR version

Applied to `example_paper.tex` / `example_paper.bib` in this folder (and to the ICLR package in `../iclr2027/`); `example_paper.pdf` and `original_vs_updated.diff` regenerated.

1. **Constrained-parameter count (§4).** "roughly 24M (40%)" → "25M (42%)": 12 blocks × (2048×512 + 512×2048) = 25,165,824 weights of T5-small's 60.5M (24M was the mebi-unit figure, 24 × 2²⁰).
2. **XSUM significance bound (§6.1).** "p ≤ 1.3 × 10⁻⁶" → "p < 1.4 × 10⁻⁶": the largest Bonferroni-corrected p is 1.32 × 10⁻⁶ (seed 8888, ROUGE-Lsum), so the old inequality was violated by rounding.
3. **Citation attribution (Appendix, related work).** Sill (1998) proves universal approximation for min–max networks of sign-constrained linear units, not for nonnegative-weight sigmoidal networks (that is Daniels & Velikova, 2010); the sentence now attributes each result to its paper.
4. **Bibliography.** `smith1995monotone` gained its series (`Mathematical Surveys and Monographs`, vol. 41) and the full publisher name, removing the only BibTeX warning ("number but no series") and the dangling "Number 41." in the reference list.
