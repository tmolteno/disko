#!/usr/bin/python3

import argparse
import datetime
import json
import logging
import sys
import os

import numpy as np
import matplotlib.pyplot as plt

from tart_tools import api_handler
from tart.imaging import elaz

from importlib.metadata import version

from disko import fov

logger = logging.getLogger(__name__)


def get_source_list(source_json, el_limit, jy_limit):
    src_list = []
    if source_json is not None:
        src_list = elaz.from_json(
            source_json, el_limit=el_limit, jy_limit=jy_limit)
    return src_list


def main():
    parser = argparse.ArgumentParser(description='Draw and visualize discrete fields of view',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('filename', help="The HDF5 field of view file")

    parser.add_argument('--catalog', required=False,
                        default='https://tart.elec.ac.nz/catalog', help="Catalog API URL.")

    parser.add_argument('--show-sources', action="store_true",
                        help="Show known sources on images (only works on PNG & SVG).")
    parser.add_argument('--elevation',
                        type=float, default=20.0,
                        help="Elevation limit for displaying sources (degrees).")

    parser.add_argument('--min',
                        type=float, default=-1,
                        help="Lower end of image range (-1 means calculate)")
    parser.add_argument('--max',
                        type=float, default=-1,
                        help="Upper end of image range. (-1 means calculate)")

    parser.add_argument('--PNG', default=None,
                        help="Generate a PNG format image.")
    parser.add_argument('--PDF', default=None,
                        help="Generate a PDF format image.")
    parser.add_argument('--SVG', default=None,
                        help="Generate a SVG format image.")
    parser.add_argument('--VTK', default=None,
                        help="Generate a VTK mesh format image.")
    parser.add_argument('--FITS', default=None,
                        help="Generate a FITS format image.")

    parser.add_argument('--sqrt', action="store_true",
                        help="Scale pixels to the square root")
    parser.add_argument('--display', action="store_true",
                        help="Display the field of view")
    parser.add_argument('--version', action="store_true",
                        help="Display the current version")
    parser.add_argument('--debug', action="store_true",
                        help="Display debugging information")

    ARGS = parser.parse_args()

    if ARGS.debug:
        level = logging.DEBUG
    else:
        level = logging.ERROR

    logger = logging.getLogger('disko')
    logger.setLevel(level)

    if ARGS.debug:
        fh = logging.FileHandler('disko_draw.log')
        fh.setLevel(level)

        # create console handler and set level to debug
        ch = logging.StreamHandler()
        ch.setLevel(level)

        # create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # add formatter to ch
        ch.setFormatter(formatter)
        fh.setFormatter(formatter)

        # add ch to logger
        logger.addHandler(ch)

    if ARGS.version:
        version = version("disko")
        print(f"disko_draw: Version {version}")
        print("       (c) 2022-2023 Tim Molteno")
        sys.exit(0)

    field_of_view = fov.from_hdf(ARGS.filename)
    ts = field_of_view.timestamp
    geo = field_of_view.geolocation
    logger.info(f"geo {geo}")
   
    if ARGS.sqrt:
        new_pixels = np.sqrt(field_of_view.pixels)
        field_of_view.set_visible_pixels(new_pixels)
 
    if ARGS.version:
        version = version("disko")
        print(f"disko_draw: Version {version}")
        print("             (c) 2023 Tim Molteno")
        sys.exit(0)

    src_list = None
    if ARGS.show_sources:
        api = api_handler.APIhandler("")
        cat_url = api.catalog_url(lon=geo.lon.to_value('deg'),
                                  lat=geo.lat.to_value('deg'),
                                  datestr=ts.isoformat())
        logger.info(f"catalog url: {cat_url}")
        source_json = api.get_url(cat_url)
        src_list = get_source_list(source_json, el_limit=ARGS.elevation, jy_limit=1e4)
        # print("Source list")
        # for s in src_list:
        #     print(f"   src: el:{np.degrees(s.el_r) :5.2f}")


    if ARGS.max == -1:
        max_img = np.max(field_of_view.pixels)
    else:
        max_img = ARGS.max

    if ARGS.min == -1:
        min_img = np.min(field_of_view.pixels)
    else:
        min_img = ARGS.min

    field_of_view.set_plot_range(min_img, max_img)

    if ARGS.VTK:
        field_of_view.write_mesh(ARGS.VTK)

    if ARGS.FITS:
        field_of_view.to_fits(fname=ARGS.FITS, info={'creator': 'disko_draw'})

    if ARGS.SVG:
        field_of_view.to_svg(fname=ARGS.SVG, src_list=src_list, show_grid=True, title=None)
        print(f"Writing SVG: {ARGS.SVG}")

    if ARGS.PNG:
        fname = ARGS.PNG
        print(f"Writing PNG: {fname}")
        plt.figure()
        field_of_view.plot(plt, None)
        plt.tight_layout()
        plt.savefig(fname, dpi=300)
        plt.close()

    if ARGS.PDF:
        field_of_view.plot(plt, None)
        plt.savefig(ARGS.PDF, dpi=600)
        plt.close()

    if ARGS.display:
        field_of_view.plot(plt, None)
        plt.show()
