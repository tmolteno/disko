#
# Copyright Tim Molteno 2017-2022 tim@elec.ac.nz
#

import json
import logging

import healpy as hp
import matplotlib.pyplot as plt
import numpy as np

from scipy import misc

from .healpix_sphere import HealpixSphere
from .disko import DiSkO
from .telescope_operator import TelescopeOperator

logger = logging.getLogger(__name__)
logger.addHandler(
    logging.NullHandler()
)  # Add other handlers if you're using this as a library
logger.setLevel(logging.INFO)


def mask_to_sky(mask, nside):
    height, width, col = mask.shape
    mask = mask / np.max(mask)

    rmax = min(width, height) / 2

    x0 = width / 2
    y0 = height / 2

    ## Scan through healpix angles (for an nside) and find out the corresponding pixel angle.
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


from argparse import ArgumentParser

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

    mask = misc.imread(ARGS.mask)
    s = mask_to_sky(mask, ARGS.nside)

    sphere = HealpixSphere(ARGS.nside)
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
