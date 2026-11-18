"""Re-implementation of the methodology in
'Cross-Asset Correlation Dynamics and Commodity Market Predictability'
(W. L. R. Talefo Taffo, SKEMA Working Paper 14(2), May 2026).

IMPORTANT: this package was written in November 2026 from the paper's text.
It is NOT the code that produced the published figures. See README.md.
"""
__version__ = "0.1.0"

import pandas as _pd
# pandas >= 2.2 uses "ME" for month-end; older versions (e.g. some Anaconda installs) use "M"
_v = tuple(int(x) for x in _pd.__version__.split(".")[:2])
MONTH_END = "ME" if _v >= (2, 2) else "M"
