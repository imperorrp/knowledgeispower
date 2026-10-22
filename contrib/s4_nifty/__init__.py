"""S4: NIFty top-scoring-pairs panel selection, registered as the contrib method ``nifty``.

Optional dependency: NIFty (https://github.com/PayneLab/nifty) is not on PyPI. Point ``NIFTY_PATH``
at a checkout; if it cannot be imported, nothing is registered and ``AVAILABLE`` is ``False``, so
``benchmarks/run.py --include-contrib`` and ``tests/contrib/test_s4_nifty.py`` skip it cleanly.
"""

from scp_graphfs_mcp.registry import register_method

from .nifty_select import available, nifty_rules, select, version_info

AVAILABLE = available()

if AVAILABLE:
    register_method(
        "nifty",
        "NIFty top-scoring-pairs rules (one-vs-rest on cell_type), proteins of the top rules; ignores the graph",
        select=select,
        contrib=True,
    )

__all__ = ["AVAILABLE", "available", "nifty_rules", "select", "version_info"]
