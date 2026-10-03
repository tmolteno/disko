# Copyright Tim Molteno 2022-2026 tim@elec.ac.nz
# License: GPLv3

"""
disko.fov — fields of view: the model of the sky being imaged
(issue #10 Phase 3).

- :mod:`disko.fov.fov` — the :class:`FoV` base class and the simple
  grid (:class:`SquareFoV`), plus the coordinate helpers and geometry
  value types they were defined with.
- :mod:`disko.fov.healpix` — the HEALPix fields of view
  (:class:`HealpixFoV`, :class:`HealpixSubFoV`).
- :mod:`disko.fov.mesh` — the unstructured adaptive mesh
  (:class:`AdaptiveMeshFoV`).
- :mod:`disko.fov.factory` — :func:`from_hdf`, the HDF loader.

The imaging algorithms live in :mod:`disko.image`.
"""

from .factory import from_hdf  # noqa: F401
from .fov import (  # noqa: F401
    ElAz,
    FoV,
    GeoLocation,
    HpAngle,
    LonLat,
    PlotCoords,
    SquareFoV,
    elaz2hp,
    elaz2lmn,
    hp2elaz,
    image_stats,
    lonlat,
)
from .healpix import HealpixFoV, HealpixSubFoV, create_fov  # noqa: F401
from .mesh import AdaptiveMeshFoV, area, get_lmn, get_mesh  # noqa: F401
