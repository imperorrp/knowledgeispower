"""S4 contract: the ``nifty`` contrib method is deterministic, returns k unique candidates, ignores the graph.

NIFty is an optional dependency (not on PyPI): set ``NIFTY_PATH`` to a clone of
https://github.com/PayneLab/nifty. Without it the contract tests skip and only the
"skips cleanly" test runs.
"""
import networkx as nx
import numpy as np
import pytest

from scp_graphfs_mcp.datasets import from_log2_matrix
from scp_graphfs_mcp.methods import candidates
from scp_graphfs_mcp.registry import METHODS, load_contrib

load_contrib()
import contrib.s4_nifty as s4  # noqa: E402  (after load_contrib so registration state is final)
from contrib.s4_nifty import nifty_select  # noqa: E402

needs_nifty = pytest.mark.skipif(not s4.AVAILABLE, reason="NIFty not importable (set NIFTY_PATH)")


@pytest.fixture
def toy():
    """60 cells in 3 classes, 30 proteins; P0..P5 separate the classes, P29 is not a candidate."""
    rng = np.random.default_rng(0)
    n, p = 60, 30
    y = np.repeat(["A", "B", "C"], n // 3)
    X = rng.normal(10, 1, (n, p))
    X[y == "A", 0] += 4; X[y == "A", 1] -= 4
    X[y == "B", 2] += 4; X[y == "B", 3] -= 4
    X[y == "C", 4] += 4; X[y == "C", 5] -= 4
    X[rng.random((n, p)) < 0.1] = np.nan  # real missingness, below the 20% candidate threshold
    X[:40, 29] = np.nan  # P29: 33% detection -> never a candidate
    import pandas as pd
    obs = pd.DataFrame({"cell_type": y}, index=[f"c{i}" for i in range(n)])
    a = from_log2_matrix(X, obs.index, [f"P{i}" for i in range(p)], obs=obs)
    g = nx.Graph(); g.add_nodes_from(a.var_names)
    g.add_edges_from([("P0", "P1"), ("P7", "P8")], weight=1.0)
    g.graph["provenance"] = {"source": "toy", "version": "0", "url": "-", "sha256": "-"}
    return a, g


def test_skips_cleanly_without_nifty(monkeypatch):
    """A bogus NIFTY_PATH must mean 'unavailable', never an exception at import or call time."""
    monkeypatch.setenv("NIFTY_PATH", "/nonexistent/nifty")
    monkeypatch.setattr(nifty_select, "_nifty", None)
    assert nifty_select.available() is False
    with pytest.raises(RuntimeError, match="NIFTY_PATH"):
        nifty_select.nifty_rules(from_log2_matrix(np.ones((3, 3)), list("abc"), list("xyz")), 2)
    monkeypatch.setattr(nifty_select, "_nifty", None)  # let the real import run again afterwards


def test_registration_matches_availability():
    assert ("nifty" in METHODS) == s4.AVAILABLE
    if s4.AVAILABLE:
        e = METHODS["nifty"]
        assert e.contrib and e.select is not None and e.impute is None


def test_vectorize_pairs_matches_nifty_semantics():
    """NaN behaves as -inf, comparison is strict, dtype int8 (DataTransformer.vectorize_all_pairs)."""
    X = np.array([[1.0, 2.0, np.nan], [3.0, 3.0, 0.5], [np.nan, np.nan, 1.0]])
    pairs = np.array([[0, 1], [0, 2], [1, 2]])
    M = nifty_select.vectorize_pairs(X, pairs, chunk=2)
    assert M.dtype == np.int8 and M.shape == (3, 3)
    assert M.tolist() == [[0, 0, 0], [1, 1, 0], [1, 1, 0]]
    if s4.AVAILABLE:
        import pandas as pd
        dt = nifty_select._import_nifty()["DataTransformer"]()
        ref = dt.vectorize_all_pairs([("a", "b"), ("a", "c"), ("b", "c")], pd.DataFrame(X, columns=list("abc")))
        assert np.array_equal(M, ref)


@needs_nifty
def test_returns_k_unique_candidates(toy):
    a, g = toy
    names = list(a.var_names)
    cand = set(np.array(names)[candidates(a)])
    for k in (4, 10, 25):
        panel = s4.select(a, g, k)
        assert len(panel) == k and len(set(panel)) == k
        assert all(isinstance(p, str) and p in names for p in panel)
        assert set(panel) <= cand, "only proteins detected in >= 80% of cells"
        assert "P29" not in panel


@needs_nifty
def test_deterministic_and_graph_independent(toy):
    a, g = toy
    nifty_select._cache.clear()
    p1 = s4.select(a, g, 10)
    nifty_select._cache.clear()
    p2 = s4.select(a, None, 10)
    nifty_select._cache.clear()
    p3 = s4.select(a, nx.Graph(), 10)
    assert p1 == p2 == p3


@needs_nifty
def test_top_rules_find_the_planted_markers(toy):
    a, _ = toy
    panel = s4.select(a, None, 6)
    assert set(panel) == {"P0", "P1", "P2", "P3", "P4", "P5"}


@needs_nifty
def test_rules_api(toy):
    a, _ = toy
    res = nifty_select.nifty_rules(a, 6)
    assert res["classes"] == ["A", "B", "C"] and res["k_rules"] == 2
    for c in res["classes"]:
        top, summary = res["top"][c], res["summary"][c]
        assert len(top) == 2 and set(top.columns) >= {"Gene_Pair", "True_Score", "P_Value"}
        assert (summary["P_Value"].diff().dropna() >= 0).all(), "summary is sorted by p-value"
        assert ((summary["True_Score"] >= 0) & (summary["True_Score"] <= 1)).all()


@needs_nifty
def test_fills_when_rules_cannot_supply_k():
    """Tiny data: 4 candidates -> 6 rules; k=4 is reachable, k beyond the candidate pool is capped."""
    import pandas as pd
    rng = np.random.default_rng(1)
    X = rng.normal(10, 1, (30, 5)); X[:, 4] = np.nan
    obs = pd.DataFrame({"cell_type": np.repeat(["A", "B"], 15)}, index=[f"c{i}" for i in range(30)])
    a = from_log2_matrix(X, obs.index, list("PQRST"), obs=obs)
    panel = s4.select(a, None, 4)
    assert sorted(panel) == list("PQRS")
    assert s4.select(a, None, 10) == panel  # cannot exceed the candidate pool; never pads with non-candidates


def test_missing_labels_raise_clearly():
    a = from_log2_matrix(np.ones((3, 3)), list("abc"), list("xyz"))
    if not s4.AVAILABLE:
        pytest.skip("NIFty not importable (set NIFTY_PATH)")
    with pytest.raises(ValueError, match="cell_type"):
        nifty_select.nifty_rules(a, 2)
