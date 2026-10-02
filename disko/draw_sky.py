#
# Copyright Tim Molteno 2022-2026 tim@elec.ac.nz
#

import logging
import os
from argparse import ArgumentParser

import healpy as hp
import imageio
import matplotlib.pyplot as plt
import numpy as np

from .healpix_sphere import HealpixFoV

logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)


def output_path(out_dir, ending, image_title):
    '''
        Path of an output image "<out_dir>/<image_title>.<ending>",
        creating the output directory on the way (issue #10, Phase 2:
        this was disko/cli.py's local path() helper).
    '''
    os.makedirs(out_dir, exist_ok=True)
    fname = "{}.{}".format(image_title, ending)
    return os.path.join(out_dir, fname)


def save_images(fov, out_dir, image_title, source_list=None, info=None,
                vtk=False, fits=False, svg=False, png=False, pdf=False,
                display=False):
    '''
        Write a FoV's images in the requested formats, titled
        <image_title> inside `out_dir` (issue #10, Phase 2: the drawing
        that used to live in disko/cli.py's save_images(), moved here so
        every CLI shares one implementation; draw_cli.py keeps its own
        explicit-filename flags and calls FoV.to_svg()/plot() directly).

        Formats and output are byte-for-byte what the CLI produced
        before: PNG at dpi=300 with a tight layout, PDF at dpi=600
        (no tight layout), SVG with the grid overplotted, FITS via
        FoV.to_fits (info = the caller's world coordinate system, which
        wins over the derived one), VTK mesh, and an optional interactive
        display.
    '''
    if vtk:
        fov.write_mesh(output_path(out_dir, "vtk", image_title))

    if fits:
        # Save as a FITS file
        fov.to_fits(fname=output_path(out_dir, "fits", image_title), info=info)

    if svg:
        fname = output_path(out_dir, "svg", image_title)
        fov.to_svg(
            fname=fname, show_grid=True, src_list=source_list, title=image_title
        )
        logger.info("Generating {}".format(fname))

    if png:
        fname = output_path(out_dir, "png", image_title)
        fov.plot(plt, source_list)
        plt.title(image_title)
        plt.tight_layout()
        plt.savefig(fname, dpi=300)
        plt.close()
        logger.info("Generating {}".format(fname))

    if pdf:
        fname = output_path(out_dir, "pdf", image_title)
        fov.plot(plt, source_list)
        plt.title(image_title)
        plt.savefig(fname, dpi=600)
        plt.close()
        logger.info("Generating {}".format(fname))

    if display:
        fov.plot(plt, source_list)
        plt.title(image_title)
        plt.show()


def mask_to_sky(mask, nside):
    height, width, col = mask.shape
    mask = mask / np.max(mask)

    rmax = min(width, height) / 2

    x0 = width / 2
    y0 = height / 2

    # Scan through healpix angles (for an nside) and find out the corresponding pixel angle.
    npix = hp.nside2npix(nside)

    pixel_indices = range(npix)
    theta, phi = hp.pix2ang(nside, pixel_indices)
    s = np.zeros(npix)

    for i in pixel_indices:
        th = theta[i]  # elevation np.pi/2 is horizon, zero vertical
        ph = phi[i]

        # Calcular image pixel corresponding to the theta, phi

        if th < np.pi / 2:
            r = rmax * np.sin(th)

            x = int(x0 + r * np.sin(ph))
            y = int(y0 + r * np.cos(ph))

            s[i] = 1.0 - np.mean(mask[y, x, :])
    return s


if __name__ == "__main__":
    import argparse

    parser = ArgumentParser(
        description="Draw something in the Null Space.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--mask", default="batman.jpeg", help="Use the mask file.")
    parser.add_argument("--nside", default=32, type=int, help="Use the mask file.")

    source_json = None

    ARGS = parser.parse_args()

    mask = imageio.v3.imread(ARGS.mask)
    s = mask_to_sky(mask, ARGS.nside)

    sphere = HealpixFoV(ARGS.nside)
    sphere.set_visible_pixels(s, scale=False)

    rot = (0, 90, 0)
    plt.figure()  # (figsize=(6,6))
    logger.info("sphere.pixels: {}".format(sphere.pixels.shape))
    if True:
        hp.orthview(
            sphere.pixels, rot=rot, xsize=1000, cbar=True, half_sky=True, hold=True
        )
        hp.graticule(verbose=False)
        plt.tight_layout()
    else:
        hp.mollview(sphere.pixels, rot=rot, xsize=1000, cbar=True)
        hp.graticule(verbose=True)

    plt.show()
