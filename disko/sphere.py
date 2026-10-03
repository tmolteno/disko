# Copyright Tim Molteno 2022-2026 tim@elec.ac.nz
# License: GPLv3

"""
Compatibility shim for the issue #10 Phase 3 package split.

The field-of-view classes moved to :mod:`disko.fov.fov`. Everything this
module used to define is re-exported from there, so every existing import
path keeps working::

    from disko.sphere import FoV, SquareFoV, GeoLocation, elaz2lmn
    from disko import sphere

New code should import from :mod:`disko.fov` instead. The shim is silent:
see CHANGES.md for why it does not emit a DeprecationWarning.
"""

from .fov.fov import *  # noqa: F401,F403
