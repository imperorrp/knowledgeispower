"""S4: NIFty top-scoring-pairs (TSP) feature selection as a panel-selection baseline.

NIFty ("Never Impute Features", Nitz, Echarry, McGee & Payne, bioRxiv 2026,
doi:10.64898/2026.03.06.710179; code https://github.com/PayneLab/nifty) scores every protein
*pair* (A, B) by how well the rule "A > B in this cell" separates two cell classes, estimates a
per-rule p-value against label-shuffled nulls bucketed by rule prevalence, and keeps the best
``k_rules`` rules after a mutual-information redundancy filter. Missing values never get imputed:
``NaN`` is treated as ``-inf`` so a missing protein never wins a comparison.

This wrapper turns NIFty's rule list into a k-protein panel so it can sit next to ``hvp``,
``random`` and the prior-propagated panels in ``benchmarks/run.py``. It needs cell labels
(``adata.obs['cell_type']``); it ignores the graph argument entirely.

NIFty is not a pip package (its modules are top-level files in the repository checkout), so it is
an *optional* dependency: point ``NIFTY_PATH`` at a clone of PayneLab/nifty, or put that
directory on ``sys.path``. If it cannot be imported, :func:`available` is ``False`` and nothing is
registered (``contrib/s4_nifty/__init__.py``), so the benchmark and the test suite skip it cleanly.

Design choices (see README.md for the reasoning):

* Candidate pool = proteins detected in >= 80% of cells (``scp_graphfs_mcp.methods.candidates``),
  the same pool ``hvp``, ``random`` and ``soft_weight`` draw from, so panels are comparable.
* Multi-class labels are handled one-vs-rest per class (NIFty's own CLI is binary: first label vs
  the rest). Each class gets ``ceil(k / n_classes)`` rules with NIFty's default MI filter
  (``mutual_information=True, cutoff 0.7, disjoint=False``); classes are merged round-robin.
* A rule contributes both of its proteins. If the merged set is short of k, it is topped up from
  the remaining NIFty-ranked rules (p-value, then score), and only then from the highest-variance
  candidates (``hvp_scores``), with the number of fills logged.
* The pair matrix is built in chunks with exactly NIFty's semantics (``NaN -> -inf``, strict ``>``,
  ``int8``) because ``DataTransformer.vectorize_all_pairs`` materialises two float64 copies of the
  (n_pairs x n_cells) matrix, which is ~2 GB for ~1000 candidates. Everything downstream
  (identical-rule removal, scoring, bucketed nulls, p-values, MI filter) is NIFty's own code.
* Fixed seed (``seed + class_index``), NIFty's own RNG; results are cached per (data, k) because
  the benchmark calls ``select`` once per prior and the output cannot depend on the prior.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import math
import os
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from scp_graphfs_mcp.methods import _top_k, candidates, hvp_scores

try:  # keep the hard resource guard in front of the dense pair matrix, as the core does
    from scp_prep import check_resources
except ImportError:  # pragma: no cover - scp_prep is a core dependency
    check_resources = None

LABEL_COLUMN = "cell_type"
MIN_DETECTION = 0.8
DEFAULT_SEED = 0
# NIFty's config.toml defaults for find_features (docs/input_file_formats.md in PayneLab/nifty)
MUTUAL_INFORMATION = True
MUTUAL_INFORMATION_CUTOFF = 0.7
DISJOINT = False
NIFTY_MODULES = ("Colors", "DataTransformer", "EvaluateRules")

_nifty = None
_cache: dict[tuple, list[str]] = {}


# --------------------------------------------------------------------------- optional import


def nifty_path() -> Path | None:
    """Directory holding the NIFty checkout, from ``NIFTY_PATH`` (or ``None``)."""
    p = os.environ.get("NIFTY_PATH")
    return Path(p).expanduser() if p else None


def _import_nifty():
    """Import NIFty's modules once; return a namespace or ``None`` if unavailable."""
    global _nifty
    if _nifty is not None:
        return _nifty or None
    p = nifty_path()
    if p is not None:
        if not (p / "EvaluateRules.py").is_file():
            _nifty = False
            return None
        if str(p) not in sys.path:
            sys.path.append(str(p))  # append, never prepend: NIFty's top-level names must not shadow ours
    try:
        import EvaluateRules  # noqa: N813  (NIFty module names are CapWords)
        import DataTransformer  # noqa: N813
    except Exception:  # noqa: BLE001  (ImportError, or statsmodels missing inside NIFty)
        _nifty = False
        return None
    _nifty = {"EvaluateRules": EvaluateRules.EvaluateRules,
              "DataTransformer": DataTransformer.DataTransformer,
              "file": Path(EvaluateRules.__file__).resolve()}
    return _nifty


def available() -> bool:
    return _import_nifty() is not None


def version_info() -> dict:
    """Where NIFty was loaded from (for provenance in results)."""
    ns = _import_nifty()
    return {"available": ns is not None, "path": str(ns["file"].parent) if ns else None}


# --------------------------------------------------------------------------- pair matrix


def vectorize_pairs(X: np.ndarray, pairs: np.ndarray, chunk: int = 200_000) -> np.ndarray:
    """``int8`` matrix (n_pairs x n_cells): 1 where ``X[:, a] > X[:, b]``, NaN counts as ``-inf``.

    Same semantics as NIFty's ``DataTransformer.vectorize_all_pairs`` (``fillna(-inf)`` then strict
    ``>``), built in chunks to avoid its two float64 copies of the pair matrix.
    """
    Xf = np.where(np.isnan(X), -np.inf, X)
    out = np.empty((len(pairs), X.shape[0]), dtype=np.int8)
    for s in range(0, len(pairs), chunk):
        a, b = pairs[s:s + chunk, 0], pairs[s:s + chunk, 1]
        out[s:s + chunk] = (Xf[:, a] > Xf[:, b]).T
    return out


def _rank_rules(ev, bool_matrix: np.ndarray, pairs: list[tuple[str, str]], labels: np.ndarray,
                k_rules: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """NIFty's ``EvaluateRules.run_rule_evaluator`` minus the TSV write and its vectoriser.

    Returns ``(summary, top)``: every surviving rule with its score, bucket and p-value, and the
    ``k_rules`` kept after NIFty's redundancy filter. Steps and order follow NIFty verbatim.
    """
    meta = pd.DataFrame({"classification_label": labels})
    binarized = ev.binarize_labels(meta)  # first label vs rest; |TP% - FP%| is symmetric in the sign
    bool_matrix, pairs = ev.remove_identical_rules_structured_view(bool_matrix, pairs)
    true_scores = ev.evaluate_pairs(bool_matrix, binarized)
    pair_to_score = dict(zip(pairs, true_scores))
    pair_to_index = {pair: i for i, pair in enumerate(pairs)}
    null_scores = ev.get_null_scores(bool_matrix, binarized)
    bucket_to_rules, bucket_to_null = ev.bookkeeping(pairs, null_scores, bool_matrix)
    bucket_to_null = ev.expand_small_null_distributions(bucket_to_null, pair_to_index, bool_matrix,
                                                        binarized, bucket_to_rules)
    summary = ev.summarize_bucket_stats(pair_to_score, bucket_to_rules, bucket_to_null)
    with contextlib.redirect_stderr(io.StringIO()):  # NIFty prints coloured WARNINGs when short of rules
        top = ev.filter_rules(summary, pair_to_index, bool_matrix, k=k_rules, mutual_info=MUTUAL_INFORMATION,
                              mi_cutoff=MUTUAL_INFORMATION_CUTOFF, disjoint=DISJOINT)
    summary = summary.sort_values(["P_Value", "True_Score"], ascending=[True, False], kind="stable")
    return summary.reset_index(drop=True), top


# --------------------------------------------------------------------------- panel


def _fingerprint(adata, k: int, seed: int) -> tuple:
    X = np.ascontiguousarray(np.asarray(adata.X, dtype=float))
    h = hashlib.sha256()
    h.update(X.tobytes())
    h.update("\x00".join(map(str, adata.var_names)).encode())
    h.update("\x00".join(map(str, adata.obs[LABEL_COLUMN].astype(str))).encode())
    return (h.hexdigest(), int(k), int(seed))


def _round_robin(lists: list[list]) -> list:
    out = []
    for i in range(max((len(x) for x in lists), default=0)):
        out.extend(x[i] for x in lists if i < len(x))
    return out


def nifty_rules(adata, k: int, seed: int = DEFAULT_SEED, min_detection: float = MIN_DETECTION,
                log=print) -> dict:
    """Run NIFty one-vs-rest on the candidate proteins and return the ranked rule tables.

    Returns ``{"classes": [...], "k_rules": int, "top": {cls: DataFrame}, "summary": {cls: DataFrame},
    "candidates": [names]}``. Raises ``RuntimeError`` if NIFty cannot be imported.
    """
    ns = _import_nifty()
    if ns is None:
        raise RuntimeError("NIFty is not importable; set NIFTY_PATH to a clone of https://github.com/PayneLab/nifty")
    if LABEL_COLUMN not in adata.obs:
        raise ValueError(f"NIFty needs cell labels in adata.obs[{LABEL_COLUMN!r}]")
    names = np.array(adata.var_names)
    cand = candidates(adata, min_detection)
    idx = np.where(cand)[0]
    X = np.asarray(adata.X, dtype=float)[:, idx]
    y = adata.obs[LABEL_COLUMN].astype(str).to_numpy()
    classes = sorted(set(y))
    if len(classes) < 2:
        raise ValueError("NIFty needs at least two cell classes")
    pair_idx = np.array(list(combinations(range(len(idx)), 2)), dtype=np.int64).reshape(-1, 2)
    pairs = [(names[idx[a]], names[idx[b]]) for a, b in pair_idx]
    if not pairs:
        raise ValueError(f"NIFty needs >= 2 candidate proteins (detected in >= {min_detection:.0%} of cells); got {len(idx)}")
    if check_resources is not None:  # int8 matrix + NIFty's structured-view copy for de-duplication
        check_resources(shape=(len(pairs), adata.n_obs), itemsize=3, hard=True, op="nifty_rules")
    bool_matrix = vectorize_pairs(X, pair_idx)
    k_rules = max(1, math.ceil(k / len(classes)))
    # NIFty's binary runs: label c vs "rest". With 2 classes both runs rank the same rules, so run once.
    runs = classes[:1] if len(classes) == 2 else classes
    top, summary = {}, {}
    for i, c in enumerate(runs):
        ev = ns["EvaluateRules"](seed + i)
        labels = np.where(y == c, c, "rest")
        summary[c], top[c] = _rank_rules(ev, bool_matrix, pairs, labels, k_rules)
        log(f"nifty: class {c!r} vs rest, {len(pairs)} rules scored, {len(top[c])} kept (k_rules={k_rules})")
    return {"classes": runs, "k_rules": k_rules, "top": top, "summary": summary,
            "candidates": list(names[idx])}


def _proteins_from(rules: list[tuple[str, str]], chosen: list[str], seen: set[str], k: int) -> None:
    for a, b in rules:
        for p in (a, b):
            if p not in seen:
                seen.add(p)
                chosen.append(p)
            if len(chosen) >= k:
                return


def select(adata, graph=None, k: int = 50, *, seed: int = DEFAULT_SEED, min_detection: float = MIN_DETECTION,
           log=print) -> list[str]:
    """``register_method`` interface: k var_names from NIFty's top rules. ``graph`` is ignored."""
    key = _fingerprint(adata, k, seed)
    if key in _cache:
        return list(_cache[key])
    chosen, seen = [], set()
    if int(candidates(adata, min_detection).sum()) >= 2:
        res = nifty_rules(adata, k, seed, min_detection, log=log)
        # 1. NIFty's kept rules, classes interleaved so no class dominates the panel
        _proteins_from(_round_robin([list(res["top"][c]["Gene_Pair"]) for c in res["classes"]]), chosen, seen, k)
        n_from_rules = len(chosen)
        # 2. still short: next-best rules by NIFty's own ranking (p-value, score), without the MI filter
        if len(chosen) < k:
            _proteins_from(_round_robin([list(res["summary"][c]["Gene_Pair"]) for c in res["classes"]]), chosen, seen, k)
    else:
        log("nifty: fewer than 2 candidate proteins, no rules to score")
        n_from_rules = 0
    n_from_rank = len(chosen) - n_from_rules
    # 3. still short (tiny data): highest-variance candidates, as hard_filter does
    if len(chosen) < k:
        for p in _top_k(hvp_scores(adata, min_detection), candidates(adata, min_detection), list(adata.var_names), adata.n_vars):
            if p not in seen:
                seen.add(p)
                chosen.append(p)
            if len(chosen) >= k:
                break
    n_fill = len(chosen) - n_from_rules - n_from_rank
    if n_from_rank or n_fill:
        log(f"nifty: k={k}: {n_from_rules} proteins from kept rules, {n_from_rank} from lower-ranked rules, "
            f"{n_fill} HVP fills")
    _cache[key] = list(chosen)
    return list(chosen)
