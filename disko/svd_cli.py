#!/usr/bin/env python
import matplotlib
import os
if os.name == 'posix' and "DISPLAY" not in os.environ:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

import argparse
import json
import logging
from copy import deepcopy

import numpy as np
import imageio

from tart.operation import settings

from tart_tools import api_imaging
from tart.imaging import elaz

from .disko import DiSkO, vis_to_real
from .telescope_operator import TelescopeOperator
from .cli import get_source_list
from .healpix_sphere import HealpixFoV
from .draw_sky import mask_to_sky

logger = logging.getLogger()

def handle_image(args, title, time_repr, src_list=None):
    """ This function manages the output of an image, drawing sources e.t.c."""
    if time_repr is None:
        image_title = title
    else:
        image_title = '{}_{}'.format(title, time_repr)
    plt.title(image_title)
    if args.PNG:
        fname = '{}.png'.format(image_title)
        fpath = os.path.join(args.dir, fname)
        plt.savefig(fpath, dpi=300)
        logger.info("Generating {}".format(fname))
    if args.PDF:
        fname = '{}.pdf'.format(image_title)
        fpath = os.path.join(args.dir, fname)
        plt.savefig(fpath, dpi=600)
        logger.info("Generating {}".format(fname))
    if args.display:
        plt.show()
    plt.close()


def fun_plot(to, data, title):
    global ARGS
    pixels = np.zeros_like(to.sphere.pixels)

    pixels = data.reshape((len(pixels),))
    to.sphere.set_visible_pixels(pixels, scale=False)

    to.sphere.plot(plt, src_list=None)
    handle_image(ARGS, title + ARGS.title, None, src_list=None)


ARGS=None


def main():
    global ARGS
    parser = argparse.ArgumentParser(description='Generate an DiSkO Image using the web api of a TART radio telescope.',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--file', required=True, default=None, help="Snapshot observation saved JSON file (visiblities, positions and more).")
    parser.add_argument('--vis', required=False, default=None, help="Use a local JSON file containing the visibilities to create the image.")
    parser.add_argument('--dir', required=False, default='.', help="Output directory.")

    parser.add_argument('--nside', type=int, default=4, help="Healpix nside parameter for display purposes only.")

    parser.add_argument('--beam', action="store_true", help="Generate a gridless beam.")

    parser.add_argument('--elevation', type=float, default=20.0, help="Elevation limit for displaying sources (degrees)")
    parser.add_argument('--display', action="store_true", help="Display Image to the user")
    parser.add_argument('--harmonics', action="store_true", help="Display the harmonics")
    parser.add_argument('--psf', action="store_true", help="Display the sum of all harmonics")

    parser.add_argument('--PNG', action="store_true", help="Generate a PNG format image")
    parser.add_argument('--PDF', action="store_true", help="Generate a PDF format image")
    parser.add_argument('--show-sources', action="store_true", help="Show known sources on images (only works on PNG).")

    parser.add_argument('--title', required=False, default="", help="Prefix the title.")
    parser.add_argument('--mask', default="batman.png", help="Use the mask file.")

    source_json = None

    ARGS = parser.parse_args()

    logger.setLevel(logging.DEBUG)

    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # add formatter to ch
    ch.setFormatter(formatter)

    # add ch to logger
    logger.addHandler(ch)

    if ARGS.file:
        logger.info("Getting Data from file: {}".format(ARGS.file))
        # Load data from a JSON file
        with open(ARGS.file, 'r') as json_file:
            calib_info = json.load(json_file)

        info = calib_info['info']
        ant_pos = calib_info['ant_pos']
        config = settings.from_api_json(info['info'], ant_pos)

        flag_list = [] # [4, 5, 14, 22]

        original_positions = deepcopy(config.get_antenna_positions())

        gains_json = calib_info['gains']
        gains = np.asarray(gains_json['gain'])
        phase_offsets = np.asarray(gains_json['phase_offset'])
        config = settings.from_api_json(info['info'], ant_pos)

        measurements = []
        for d in calib_info['data']:
            vis_json, source_json = d
            cv, timestamp = api_imaging.vis_calibrated(vis_json, config, gains, phase_offsets, flag_list)
            src_list = elaz.from_json(source_json, 0.0)
        if not ARGS.show_sources:
            src_list = None

    time_repr = "{:%Y_%m_%d_%H_%M_%S_%Z}".format(timestamp)

    # Processing
    should_make_images = ARGS.display or ARGS.PNG or ARGS.PDF

    grid = DiSkO.from_cal_vis(cv)
    nside = ARGS.nside

    sphere = HealpixFoV(nside)
    # Now create the SVD of a telescope. First form the gamma matrix.
    to = TelescopeOperator(grid, sphere)

    # Have a look at some harmonics

    if ARGS.harmonics:
        mask = imageio.imread(ARGS.mask)
        s = mask_to_sky(mask, ARGS.nside)
        dut = to.gamma @ s
        print("Batman Vis = {}".format(dut))
        fun_plot(to, s, "batman")
        null_batman = to.sky_to_null(s)
        print(null_batman)
        fun_plot(to, null_batman, "null_space_batman")
        dut = to.gamma @ null_batman
        print("Null Batman Vis = {}".format(dut))

        for h in range(0, 4):
            harmonic = to.harmonic(h)  #
            fun_plot(to, harmonic, "fringe_{:04d}".format(h))

            har = to.range_harmonic(h)
            fun_plot(to, har, "range_fringe_{:04d}".format(h))

            har = to.range_harmonic(to.n_r() - h - 1)
            fun_plot(to, har, "range_fringe_{:04d}".format(to.n_r() - h - 1))

            har = to.natural_A_row(h)
            fun_plot(to, to.natural_to_sky(har), "natural_A_fringe_{:05d}".format(h))

            null_harmonic = to.null_harmonic(h)
            fun_plot(to, (null_harmonic), "natural_null_fringe_{:05d}".format(h))

            h2 = to.n_n() - h - 1
            null_harmonic = to.null_harmonic(h2)
            fun_plot(to, (null_harmonic), "natural_null_fringe_{:05d}".format(h2))

            #harmonic = Vh @ null_harmonic 
            #fun_plot(harmonic, "natural_restored_fringe_{}".format(h))

    if ARGS.psf:
        # Make a point sky
        sky = np.zeros((to.n_s, 1))
        sky[to.n_s-100] = 1.0
        sky = sky.reshape([-1,1])

        vis = to.gamma @ sky 

        sky = to.image_visibilities(vis, sphere)
        sphere.plot(plt, src_list)
        handle_image(ARGS, "psf_natural" + ARGS.title, time_repr, src_list)

        alpha = 0.1
        sky = to.image_tikhonov(vis, sphere, alpha)
        sphere.plot(plt, src_list)
        handle_image(ARGS, "psf_tikhonov_{:04.2f}".format(alpha) + ARGS.title, time_repr, src_list)

        pixels = grid.image_visibilities(vis, sphere)
        sphere.plot(plt, src_list)
        handle_image(ARGS, "psf_gridless" + ARGS.title, time_repr, src_list)

    if should_make_images:
        if ARGS.show_sources:
            src_list = get_source_list(source_json, el_limit=ARGS.elevation, jy_limit=1e4)

        real_vis = vis_to_real(grid.vis_arr)

        sky = to.image_visibilities(real_vis, sphere)
        sphere.plot(plt, src_list)
        handle_image(ARGS, "natural" + ARGS.title, time_repr, src_list)

        for i in np.linspace(0.0, 1.0, 11):
            sky = to.image_tikhonov(real_vis, sphere, float(i))
            sphere.plot(plt, src_list)
            handle_image(ARGS, "tikhonov_{:04.2f}".format(i) + ARGS.title, time_repr, src_list)
            pixels = grid.image_lasso(grid.vis_arr, sphere, float(i/5), scale=True)
            sphere.plot(plt, src_list)
            handle_image(ARGS, "lasso_{:04.2f}".format(i) + ARGS.title, time_repr, src_list)

        pixels = grid.image_visibilities(grid.vis_arr, sphere)
        sphere.plot(plt, src_list)
        handle_image(ARGS, "gridless" + ARGS.title, time_repr, src_list)

    if ARGS.beam:
        grid.beam(plt, nside)
        handle_image(ARGS, "dirty", "beam", src_list)
