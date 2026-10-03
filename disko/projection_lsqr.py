# Copyright Tim Molteno 2022-2026 tim@elec.ac.nz
# License: GPLv3

"""
Compatibility shim for the issue #10 Phase 3 package split.

The subspace-projection LSQR solver moved to
:mod:`disko.image.projection_lsqr`. Everything this module used to define
is re-exported from there, so every existing import path keeps working::

    from disko.projection_lsqr import plsqr

New code should import from :mod:`disko.image` instead. The shim is
silent: see CHANGES.md for why it does not emit a DeprecationWarning.
"""

from .image.projection_lsqr import *  # noqa: F401,F403
