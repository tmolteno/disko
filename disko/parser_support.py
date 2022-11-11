import argparse


from .resolution import Resolution
from .healpix_sphere import create_fov
from .sphere_mesh import AdaptiveMeshSphere


def sphere_add_args(parser):
    
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument('--fov', type=str, default="180deg",
                        help="Field of view. E.g. 1.3deg, 12\", 11', 8uas, 6mas...")
    parent_parser.add_argument('--res', type=str, default="2deg",
                        help="Maximim Resolution of the sky. E.g. 1.3deg, 12\", 11', 8uas, 6mas.")

    sphere_subparsers = parser.add_subparsers(dest='sphere_type')
    mesh_parser = sphere_subparsers.add_parser('--mesh',
                                        help='Use a non-structured mesh in the image space',
                                        parents = [parent_parser])
    mesh_parser.add_argument('--adaptive', type=int, default=0,
                             help="Use N cycles of adaptive meshing")
    mesh_parser.add_argument('--res-min', type=str, default=None,
                             help="Highest allowed res of the sky. E.g. 1.3deg, 12\", 11', 8uas, 6mas.")
    
    parser_healpix = sphere_subparsers.add_parser('--healpix', 
                                                  help='Use HealPix tiling',
                                                  parents=[parent_parser])
    parser_healpix.add_argument('--nside', type=int, default=None, help="Healpix nside parameter")

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
