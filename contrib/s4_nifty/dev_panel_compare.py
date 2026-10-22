"""S4 dev comparison: NIFty panels on the phase-2 panel harness (dev CV seeds 100-102 only).

Scores ``hvp``, ``random__0..9`` and ``nifty`` exactly as ``benchmarks/explore_panel.py`` does
(``run.panel_r2`` on kNN-filled Petrosius 2023, targets detected in >= 50% of cells), checks the
re-scored hvp/random rows against the stored ``results/phase2/dev/dev_panel.json``, and merges in
the stored coverage / pivoted-QR / data-matched rows so NIFty sits next to them. Read-only on
``results/``; writes one CSV.

    NIFTY_PATH=/path/to/nifty python contrib/s4_nifty/dev_panel_compare.py \
        [--out contrib/s4_nifty/dev_panel_means.csv]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmarks"))
from run import KS, panel_r2  # noqa: E402

from scp_graphfs_mcp import phase2 as P  # noqa: E402
from scp_graphfs_mcp.complexes import in_any_complex  # noqa: E402
from scp_graphfs_mcp.datasets import load_petrosius2023  # noqa: E402
from scp_graphfs_mcp.ids import group_accessions  # noqa: E402
from scp_graphfs_mcp.methods import knn_fill, select_features, select_random  # noqa: E402

from contrib.s4_nifty import nifty_select as ns  # noqa: E402

DEV_SEEDS = [100, 101, 102]  # dev only; never the confirmation seeds
STORED = ROOT / "results" / "phase2" / "dev" / "dev_panel.json"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "benchmarks" / "_quick_out" / "s4_dev_panel_means.csv")
    a = ap.parse_args(argv)
    if not ns.available():
        sys.exit("NIFty not importable; set NIFTY_PATH to a clone of https://github.com/PayneLab/nifty")
    adata = load_petrosius2023()
    names = list(adata.var_names)
    X = np.asarray(adata.X, float)
    Xknn = knn_fill(adata)
    in_cx = np.array([in_any_complex(group_accessions(f)) for f in names])
    target = (~np.isnan(X)).mean(0) >= 0.5
    kmask = P.keratin_mask(names)

    rows = []
    for k in KS:
        panels = {"hvp": select_features(adata, None, k, "none"), "nifty": ns.select(adata, None, k)}
        for s in range(10):
            panels[f"random__{s}"] = select_random(adata, k, s)
        for m, panel in panels.items():
            idx = [names.index(p) for p in panel]
            for s in DEV_SEEDS:
                r = panel_r2(Xknn, X, idx, target, in_cx, seed=s)
                rows.append({"k": k, "method": m, "cv_seed": s, **{f"r2_{x}": v for x, v in r.items()},
                             "n_keratin": int(kmask[idx].sum())})
    df = pd.DataFrame(rows)

    stored = pd.DataFrame(json.loads(STORED.read_text())["rows"])
    chk = df[df.method != "nifty"].merge(stored, on=["k", "method", "cv_seed"], suffixes=("", "_stored"))
    drift = float((chk.r2_all - chk.r2_all_stored).abs().max())
    print(f"re-scored hvp/random vs stored dev rows: {len(chk)} rows, max |diff| = {drift:g}")
    if drift > 1e-9:
        sys.exit("harness drift: re-scored rows differ from results/phase2/dev/dev_panel.json; not merging")

    keep = ["hvp_W1", "pivqr_W1", "cover_corum_W1", "cover_complexportal_W1",
            "cover_data_matched_corum_W1", "cover_data_matched_complexportal_W1"]
    both = pd.concat([df, stored[stored.method.isin(keep)]], ignore_index=True)
    both["arm"] = both.method.where(~both.method.str.startswith("random__"), "random")
    cols = ["r2_all", "r2_in_complex", "r2_not_in_complex", "n_keratin"]
    out = both.groupby(["k", "arm"])[cols].mean()
    rnd = both[both.arm == "random"].groupby(["k", "method"]).r2_all.mean().groupby("k")
    out["random_min_r2_all"] = rnd.min().reindex(out.index.get_level_values("k")).to_numpy()
    out["random_max_r2_all"] = rnd.max().reindex(out.index.get_level_values("k")).to_numpy()
    out["n_random_panels_ge_this"] = [
        int((both[(both.k == k) & (both.arm == "random")].groupby("method").r2_all.mean() >= v).sum())
        for (k, _), v in out.r2_all.items()]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    out.round(4).to_csv(a.out)
    print(out.round(3).to_string())
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
