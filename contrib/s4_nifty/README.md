# S4: NIFty baseline wrapper (`nifty` panel-selection method)

**What:** NIFty ("Never Impute Features"; Nitz, Echarry, McGee & Payne, bioRxiv 2026,
[10.64898/2026.03.06.710179](https://doi.org/10.64898/2026.03.06.710179); code
[PayneLab/nifty](https://github.com/PayneLab/nifty)) is a top-scoring-pairs (TSP) feature-selection
and classification pipeline for single-cell proteomics that never imputes. This slot wraps its
`find_features` step as a contrib **panel-selection** method so it sits next to `hvp`, `random` and
the prior-propagated panels in `benchmarks/run.py` (spec: `docs/specs/S4_nifty.md`).

**Why:** NIFty is the closest published, prior-free, missingness-tolerant feature selector for SCP
(README prior-art table; it already benchmarks on Petrosius data). The project's finding that
random panels beat HVP makes an extra data-only baseline valuable: a knowledge prior should have to
beat a *strong* data-driven selector, not only variance ranking.

## Install (optional dependency)
NIFty is not on PyPI; its modules are top-level files in the repository. Nothing is added to
`pyproject.toml`.

```bash
git clone https://github.com/PayneLab/nifty /path/to/nifty          # commit da3db09 (2026-05-06) was used here
export NIFTY_PATH=/path/to/nifty                                     # or put that dir on sys.path
pip install statsmodels cloudpickle                                  # NIFty's own deps not already in this env
pytest -q tests/contrib/test_s4_nifty.py                             # 14 tests; 6 need NIFty and skip without it
python benchmarks/run.py --include-contrib                           # -> results/contrib/panel.json, rows "nifty:<prior>"
python contrib/s4_nifty/dev_panel_compare.py                         # dev seeds 100-102, next to coverage panels
```
Without NIFty (as in CI), 8 tests still run. They replace NIFty's rule ranking with a transparent
stand-in and check the wrapper's own logic. With NIFty, the other 6 also check the real method,
including that our rule ranking equals NIFty's own `run_rule_evaluator` output.

Quick check: `python benchmarks/run.py --include-contrib --quick` refuses to run, because
`--include-contrib` defaults its output to `results/contrib/` and `--quick` may not write under
`results/`. Add `--out benchmarks/_quick_out/contrib`.
If NIFty cannot be imported, `contrib.s4_nifty.AVAILABLE` is `False`, nothing is registered, and the
benchmark and tests run exactly as before (the contrib tests skip with a message).

## Interface
`select(adata, graph, k) -> list[str]`, registered as `register_method("nifty", ..., select=select,
contrib=True)`. The graph is **ignored**; `benchmarks/run.py` still calls the method once per prior,
so every `nifty:<prior>` row carries the same panel (results are cached per data + k).
Requires cell labels in `adata.obs["cell_type"]`.

`nifty_rules(adata, k)` exposes the ranked rule tables (per class: NIFty's kept rules and the full
scored table with bucket p-values) for anyone who wants the rules rather than the proteins.

## What NIFty does (its code, called unchanged)
For every candidate pair (A, B), the rule "A > B in this cell" is a binary feature; missing values
count as `-inf`, so a missing protein never wins a comparison (no imputation). Each rule is scored
by `|TP% − FP%|` between the two classes, rules with identical binary patterns are collapsed, a
label-shuffled null is built per rule-prevalence bucket (expanded to >= 100 draws), p-values are
computed against it, and the best `k_rules` rules are kept after a normalised-mutual-information
redundancy filter (`mutual_information=True`, cutoff 0.7, `disjoint=False`: NIFty's defaults).
Everything from identical-rule removal to the final filter is `EvaluateRules` from PayneLab/nifty.

## Wrapper decisions (and why)
| Decision | Why |
|---|---|
| Candidate pool = proteins detected in >= 80% of cells (`methods.candidates`) | Same pool as `hvp`, `random` and `soft_weight`, so panels differ only in the ranking, not in eligibility. NIFty's own filter (<= 50% missing in every class) would admit proteins the other arms cannot pick. |
| One-vs-rest per class, `ceil(k / n_classes)` rules each, merged round-robin | NIFty's CLI is binary (first label vs rest). Petrosius has four balanced cell types; one-vs-rest keeps every class represented. With two classes a single run is made (both directions rank identically because the score is symmetric). |
| Rule -> both proteins; top up from lower-ranked rules, then HVP, and log the fills | A panel needs k *proteins*; rules can overlap. Lower-ranked rules keep NIFty's ordering; the HVP fill only triggers on tiny toy data (mirrors `contrib/hard_filter`). |
| Pair matrix built in chunks (`vectorize_pairs`) with NIFty's exact semantics | NIFty's `DataTransformer.vectorize_all_pairs` materialises two float64 (n_pairs x n_cells) copies: ~2 GB for ~1000 candidates. The int8 result is identical (tested against NIFty's function) and sits behind `scp_prep.check_resources`. |
| Fixed seed (`0 + class index`), NIFty's own RNG | Same command -> same panel (invariant). |
| `k_rules` may exceed NIFty's CLI cap of 50 | The cap lives in NIFty's `ParameterChecker` (CLI only). With four classes and k = 100 it is 25 rules per class, within the cap anyway. |

## Output
`python benchmarks/run.py --include-contrib` writes `results/contrib/panel.json` with
`nifty:<prior>` rows for k = 25, 50, 100 (R² of ridge prediction of the non-panel proteins, all /
in-complex / not-in-complex strata), plus the panels themselves under `panels`.

Run committed with this slot (`results/contrib/`, Petrosius 2023, 1007 candidates, 506,521 rules,
code commit `abb3170`, NIFty commit `da3db09`), "all" stratum, panel R²:

| k | nifty | hvp | random (mean of 10) | complexportal | complexportal null (mean of 10) | string_exp |
|---|---|---|---|---|---|---|
| 25 | 0.479 | 0.228 | 0.450 | 0.231 | 0.286 | 0.385 |
| 50 | 0.432 | 0.189 | 0.419 | 0.193 | 0.380 | 0.403 |
| 100 | 0.296 | 0.236 | 0.248 | 0.230 | 0.269 | 0.283 |

The `nifty:*` rows are identical across priors (the graph is ignored). In this run NIFty's panel is
above `random_mean` at every k (by 0.03, 0.01 and 0.05) and above every prior-propagated
(`soft_weight`) panel, in all three strata. `run.py` stores only the *mean* of the random panels,
so this table alone cannot say whether those margins exceed random-panel spread; the dev comparison
below can. The `--quick` run gives identical `nifty` values (selection uses no benchmark seed).
Selection took 20–26 s per k in this container.

### Dev comparison next to coverage panels (`dev_panel_compare.py`, CV seeds 100–102)
The S4 track note in `docs/SLOTS.md` asks for NIFty "next to random and coverage panels", which live
in the phase-2 panel harness (`benchmarks/explore_panel.py`), not in `run.py`. `dev_panel_compare.py`
re-scores hvp, the 10 random panels and NIFty with that harness, checks that the re-scored hvp/random
rows match the stored `results/phase2/dev/dev_panel.json` exactly, and merges in the stored coverage
rows. Output: `dev_panel_means.csv`. Mean panel R² ("all" stratum) over the three dev CV seeds:

| k | nifty | random mean (min–max of 10) | random panels ≥ nifty | cover_corum_W1 | cover_complexportal_W1 | cover_data_matched_corum_W1 | cover_data_matched_complexportal_W1 |
|---|---|---|---|---|---|---|---|
| 25 | 0.477 | 0.448 (0.411–0.470) | 0 of 10 | 0.452 | 0.447 | 0.487 | 0.472 |
| 50 | 0.428 | 0.416 (0.402–0.434) | 1 of 10 | 0.406 | 0.402 | 0.449 | 0.427 |
| 100 | 0.294 | 0.244 (0.197–0.261) | 0 of 10 | 0.261 | 0.237 | 0.292 | 0.294 |

Reading (dev seeds, one dataset, no confirmation): NIFty beats every random panel at k = 25 and
100, but at k = 50 it is inside the random range. It beats curated-complex coverage at every k.
It is level with, not above, the **data-matched** coverage panels (coverage over a co-variation
graph with the same edge count as the curated one): data-matched CORUM coverage is higher at k = 25
and 50. So NIFty joins data-matched coverage as the strongest data-only panel bar, and a
curated-knowledge panel would have to beat both. NIFty panels contain 0, 0 and 1 keratins at k = 25,
50 and 100, without the W1 keratin filter.

Caveat: NIFty uses the cell-type labels to choose proteins, and no other arm does. The score is
label-free (ridge R² of the other proteins), but in a dataset of four sorted cell types, proteins
that separate the types plausibly carry much of the shared variation. The comparison is fair in
that the other arms *could not* use labels. It is not an equal-information comparison.

## Limits
- Needs labels; unsupervised datasets cannot use it. `run.py`'s classification sanity check does
  not score contrib panels; if it ever does, a NIFty panel would be circular there because labels
  were used for selection.
- Dev results above are exploratory (dev seeds, one dataset) and are not a claim. Per `AGENTS.md`
  they should also be logged in `docs/EXPLORATION_LOG.md`, which is outside the contributor paths,
  so this README is the log and the PR asks Amit to add a line.
- Runtime scales with the number of candidate pairs (quadratic in candidates); NIFty's
  mutual-information filter is a Python loop over kept rules.
- NIFty's p-values come from one shuffled-label draw per rule plus bucket pooling; they rank rules,
  they are not calibrated significance.
