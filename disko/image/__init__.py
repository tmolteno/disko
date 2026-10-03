# Copyright Tim Molteno 2022-2026 tim@elec.ac.nz
# License: GPLv3

"""
disko.image — the radio-astronomical imaging algorithms (issue #10 Phase 3).

This package holds the code that maps visibilities to sky images and back:

- :mod:`disko.image.disko` — the DiSkO measurement operator and imager
  (:class:`DiSkO`, :class:`DiSkOOperator`) plus its visibility-domain
  helpers (``jomega``, ``vis_to_real``, ``get_all_uvw``).
- :mod:`disko.image.telescope_operator` — :class:`TelescopeOperator` and
  the SVD decompositions it is built on (``normal_svd``, ``dask_svd``),
  with the ``plot_spectrum``/``plot_uv`` diagnostics.
- :mod:`disko.image.projection_lsqr` — the subspace-projection LSQR
  solver (``plsqr``).

Everything else in the top-level ``disko`` package is infrastructure:
measurement-set I/O (``ms_helper``), command-line entry points, rendering
(``draw_sky``), coordinate transforms (``coords``) and generic numerics
(``util``, ``resolution``, ``rime``, ``multivariate_gaussian``). The sky
model itself lives in :mod:`disko.fov`.
"""

from .disko import (  # noqa: F401
    DiSkO,
    DiSkOOperator,
    get_all_uvw,
    jomega,
    vis_to_real,
)
from .projection_lsqr import plsqr  # noqa: F401
from .telescope_operator import (  # noqa: F401
    TelescopeOperator,
    dask_svd,
    normal_svd,
    plot_spectrum,
    plot_uv,
)
