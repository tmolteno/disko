# Copyright Tim Molteno 2022-2026 tim@elec.ac.nz
# License: GPLv3

"""
Compatibility shim for the issue #10 Phase 3 package split.

The unstructured-mesh field of view moved to :mod:`disko.fov.mesh`.
Everything this module used to define is re-exported from there, so every
existing import path keeps working::

    from disko.sphere_mesh import AdaptiveMeshFoV, get_mesh, area

New code should import from :mod:`disko.fov` instead. The shim is silent:
see CHANGES.md for why it does not emit a DeprecationWarning.
"""

from .fov.mesh import *  # noqa: F401,F403
