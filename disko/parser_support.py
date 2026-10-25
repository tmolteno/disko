#
# Copyright Tim Molteno 2022-2026 tim@elec.ac.nz
#

import argparse
import logging

import numpy as np

from .resolution import Resolution
from .fov.healpix import create_fov
from .fov.mesh import AdaptiveMeshFoV

logger = logging.getLogger("disko")


def sphere_args_parser():
    par_res = argparse.ArgumentParser(add_help=False)
    par_res.add_argument('--fov', type=str, default="180deg",
                         help="Field of view. E.g. 1.3deg, 12\", 11', 8uas, 6mas...")
    par_res.add_argument('--res', type=str, default="2deg",
                         help="Maximim Resolution of the sky. E.g. 1.3deg, 12\", 11', 8uas, 6mas.")

    par_msh = argparse.ArgumentParser(add_help=False)
    par_msh.add_argument('--mesh', action="store_true",
                         help="Use a non-structured mesh in the image space")
    par_msh.add_argument('--adaptive', type=int, default=0,
                         help="Use N cycles of adaptive meshing")
    par_msh.add_argument('--res-min', type=str, default=None,
                         help="Highest allowed res of the sky. E.g. 1.3deg, 12\", 11', 8uas, 6mas.")

    par_sph = argparse.ArgumentParser(add_help=False)
    # default=SUPPRESS keeps ArgumentDefaultsHelpFormatter from printing a
    # misleading "(default: False)": healpix *is* the default sky (issue #12),
    # it just isn't an argparse default, because --mesh opts out of it.
    par_sph.add_argument('--healpix', action="store_true", default=argparse.SUPPRESS,
                         help="Use HealPix tiling. This is the default when "
                              "--mesh is not given, so the flag is only needed "
                              "to be explicit about it.")
    par_sph.add_argument('--nside', type=int, default=None,
                         help="Healpix nside parameter for display purposes only.")

    return [par_res, par_msh, par_sph]


def sphere_from_args(args):
    fov = Resolution.from_string(args.fov)
    res = Resolution.from_string(args.res)

    # Issue #12: a bare "disko --ms x.ms" should image the field of view, not
    # stop with "Either --mesh or --healpix must be specified". With neither
    # given, use the healpix tiling (what every example recommends).
    # --healpix is default=SUPPRESS, so read it defensively and then always
    # normalise it onto the namespace (callers may inspect it afterwards).
    use_healpix = bool(getattr(args, "healpix", False))
    if not args.mesh and not use_healpix:
        logger.info("No --mesh/--healpix given: defaulting to the healpix tiling")
        use_healpix = True
    args.healpix = use_healpix

    sphere = None
    if args.mesh:
        if args.res_min is None:
            res_min = res
        else:
            res_min = Resolution.from_string(args.res_min)

        sphere = AdaptiveMeshFoV(res_min=res_min,
                                 res_max=res,
                                 theta=np.radians(0.0),
                                 phi=0.0, fov=fov)
    if args.healpix:
        sphere = create_fov(args.nside, fov=fov, res=res, theta=0.0, phi=0.0)
    if sphere is None:
        raise RuntimeError(
            "Either --mesh or --healpix must be specified (healpix recommended)")

    return sphere
