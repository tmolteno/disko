import argparse
import numpy as np

from .resolution import Resolution
from .healpix_sphere import create_fov
from .sphere_mesh import AdaptiveMeshSphere


def sphere_args_parser():
    parser_resolution = argparse.ArgumentParser(add_help=False)
    parser_resolution.add_argument('--fov', type=str, default="180deg", help="Field of view. E.g. 1.3deg, 12\", 11', 8uas, 6mas...")
    parser_resolution.add_argument('--res', type=str, default="2deg", help="Maximim Resolution of the sky. E.g. 1.3deg, 12\", 11', 8uas, 6mas.")

    parser_mesh = argparse.ArgumentParser(add_help=False)
    parser_mesh.add_argument('--mesh', action="store_true", help="Use a non-structured mesh in the image space")
    parser_mesh.add_argument('--adaptive', type=int, default=0, help="Use N cycles of adaptive meshing")
    parser_mesh.add_argument('--res-min', type=str, default=None, help="Highest allowed res of the sky. E.g. 1.3deg, 12\", 11', 8uas, 6mas.")
    
    
    parser_sphere = argparse.ArgumentParser(add_help=False)

    parser_sphere.add_argument('--healpix', action="store_true", help="Use HealPix tiling")
    parser_sphere.add_argument('--nside', type=int, default=None, help="Healpix nside parameter for display purposes only.")

    return [parser_resolution, parser_mesh, parser_sphere]


def sphere_from_args(args):
    fov = Resolution.from_string(args.fov)
    res = Resolution.from_string(args.res)

    sphere = None
    if args.mesh:
        if args.res_min is None:
            res_min = res
        else:
            res_min = Resolution.from_string(args.res_min)

        sphere = AdaptiveMeshSphere.from_resolution(res_min=res_min, res_max=res, theta=np.radians(0.0), phi=0.0, fov=fov)
    if args.healpix:
        sphere = create_fov(args.nside, fov=fov, res=res)

    if sphere is None:
        raise RuntimeError("Either --mesh or --healpix must be specified")

    return sphere
