#!/usr/bin/env python
import matplotlib
import os
if os.name == 'posix' and "DISPLAY" not in os.environ:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

import argparse
import json
import logging
import sys

import numpy as np

from importlib.metadata import version
from copy import deepcopy
from tart.operation import settings

from tart_tools import api_imaging
from tart.imaging import elaz

from .disko import DiSkO
from .parser_support import sphere_from_args, sphere_args_parser

import tart2ms
from tart2ms import get_array_location

logger = logging.getLogger(__name__)


def get_source_list(source_json, el_limit, jy_limit):
    src_list = []
    if source_json is not None:
        src_list = elaz.from_json(
            source_json, el_limit=el_limit, jy_limit=jy_limit)
    return src_list


def disko_from_ms(ms, num_vis, res, channel=0, field_id=0, ddid=0):
    u_arr, v_arr, w_arr, frequency, cv_vis, \
        hdr, tstamp, rms, indices = tart2ms.casa_read_ms(
            ms, num_vis, angular_resolution=res.degrees(),
            channel=channel,
            field_id=field_id, ddid=ddid, pol=0
        )

    # Measurement sets do not return the conjugate pairs of visibilities

    full_u_arr = np.concatenate((u_arr, -u_arr), 0)
    full_v_arr = np.concatenate((v_arr, -v_arr), 0)
    full_w_arr = np.concatenate((w_arr, -w_arr), 0)
    full_rms = np.concatenate((rms, rms), 0)
    full_cv_vis = np.concatenate((cv_vis, np.conjugate(cv_vis)), 0)

    ret = DiSkO(full_u_arr, full_v_arr, full_w_arr, frequency)
    # Natural weighting  # np.array(cv_vis, dtype=COMPLEX_DATATYPE)
    ret.vis_arr = full_cv_vis / full_rms
    ret.timestamp = tstamp
    ret.rms = full_rms
    ret.info = hdr
    ret.indices = indices

    logger.info(f"Visibilities: {ret.vis_arr.shape}")
    logger.debug(f"u,v,w: {ret.u_arr.shape}")
    return ret


def main():
    sphere_parsers = sphere_args_parser()

    parser = argparse.ArgumentParser(
        description='DiSkO: Generate an Discrete Sky Operator Image.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=sphere_parsers)

    data_group = parser.add_mutually_exclusive_group()
    data_group.add_argument('--file', required=False, default=None, help="Snapshot observation saved JSON file (visiblities, positions and more).")
    data_group.add_argument('--ms', required=False, default=None, help="visibility file")

    parser.add_argument('--nvis', type=int, default=1000, help="Number of visibilities to use.")
    parser.add_argument('--vis', required=False, default=None, help="Use a local JSON file containing the visibilities to create the image.")
    parser.add_argument('--channel', type=int, default=0, help="Use this frequency channel.")
    parser.add_argument('--field', type=int, default=0, help="Use this FIELD_ID from the measurement set.")
    parser.add_argument('--ddid', type=int, default=0, help="Use this DDID from the measurement set.")

    algo_group = parser.add_mutually_exclusive_group()
    algo_group.add_argument('--lsqr', action="store_true", help="Use lsqr in matrix-free")
    algo_group.add_argument('--lsmr', action="store_true", help="Use lsmr in matrix-free")
    algo_group.add_argument('--fista', action="store_true", help="Use FISTA in matrix-free")
    algo_group.add_argument('--lasso', action="store_true", help="Use L1 regularization.")
    algo_group.add_argument('--tikhonov', action="store_true", help="Use L2 regularization.")

    parser.add_argument('--matrix-free', action="store_true", help="Use matrix-free regularization.")
    parser.add_argument('--niter', type=int, default=100, help="Number of iterations for iterative solutions.")

    parser.add_argument('--dir', required=False, default='.', help="Output directory.")
    parser.add_argument('--alpha', type=float, default=None, help="Regularization parameter.")
    parser.add_argument('--l1-ratio', type=float, default=0.02, help="Regularization parameter, ratio of l1 to l2 (1.0 means l1 only).")

    parser.add_argument('--show-sources', action="store_true", help="Show known sources on images (only works on PNG & SVG).")
    parser.add_argument('--title', required=False, default="disko", help="Prefix the output files.")
    parser.add_argument('--elevation', type=float, default=20.0, help="Elevation limit for displaying sources (degrees).")
    parser.add_argument('--display', action="store_true", help="Display Image to the user.")
    parser.add_argument('--PNG', action="store_true", help="Generate a PNG format image.")
    parser.add_argument('--PDF', action="store_true", help="Generate a PDF format image.")
    parser.add_argument('--SVG', action="store_true", help="Generate a SVG format image.")
    parser.add_argument('--HDF', required=False, help="Generate a HDF format field of view.")
    parser.add_argument('--VTK', action="store_true", help="Generate a VTK mesh format image.")
    parser.add_argument('--FITS', action="store_true", help="Generate a FITS format image.")

    parser.add_argument('--cv', action="store_true", help="Use Cross Validation")
    parser.add_argument('--dask', action="store_true", help="Use dask")

    parser.add_argument('--debug', action="store_true", help="Display debugging information")
    parser.add_argument('--version', action="store_true", help="Display the current version")

    source_json = None

    ARGS = parser.parse_args()

    if ARGS.debug:
        level = logging.DEBUG
    else:
        level = logging.ERROR

    logging.basicConfig()
    logger = logging.getLogger('disko')
    logger.setLevel(level)
    tart2ms_log = logging.getLogger("tart2ms")
    tart2ms_log.setLevel(level)

    if ARGS.debug:
        fh = logging.FileHandler('disko.log')
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
        print(f"disko: Version {version}")
        print("       (c) 2022-2023 Tim Molteno")
        sys.exit(0)

    sphere = sphere_from_args(ARGS)

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
        disko = DiSkO.from_cal_vis(cv)

        lat = config.get_lat()
        lon = config.get_lon()
        height = config.get_alt()

    elif ARGS.ms:
        logger.info(f"Getting Data from MS file: {ARGS.ms} to {sphere}")

        if not os.path.exists(ARGS.ms):
            raise RuntimeError(f"Measurement set {ARGS.ms} not found")

        min_res = sphere.min_res()
        logger.info(f"Min Res {min_res}")
        disko = disko_from_ms(ARGS.ms, ARGS.nvis, res=min_res, channel=ARGS.channel, field_id=ARGS.field, ddid=ARGS.ddid)
        # Convert from reduced Julian Date to timestamp.
        timestamp = disko.timestamp

        json_info = get_array_location(ARGS.ms)
        lat = json_info['lat']
        lon = json_info['lon']
        height =json_info['height']

    else:
        logger.info("Getting Data from API: {}".format(ARGS.api))

        api = api_handler.APIhandler(ARGS.api)
        config = api_handler.get_config(api)

        gains = api.get('calibration/gain')

        if (ARGS.vis is None):
            vis_json = api.get('imaging/vis')
        else:
            with open(ARGS.vis, 'r') as json_file:
                vis_json = json.load(json_file)

        ts = api_imaging.vis_json_timestamp(vis_json)
        if ARGS.show_sources:
            cat_url = api.catalog_url(lon=config.get_lon(),
                                      lat=config.get_lat(),
                                      datestr=ts.isoformat())
            source_json = api.get_url(cat_url)

        logger.info("Data Download Complete")

        cv, timestamp = api_imaging.vis_calibrated(vis_json, config, gains['gain'], gains['phase_offset'], flag_list=[])
        disko = DiSkO.from_cal_vis(cv)

        lat = config.get_lat()
        lon = config.get_lon()
        height = config.get_alt()

    sphere.set_info(timestamp=timestamp,
                    lon=lon, lat=lat, height=height)
    
    if not ARGS.show_sources:
        src_list = None
    # api_imaging.rotate_vis(ARGS.rotation, cv, reference_positions = deepcopy(config.get_antenna_positions()))
    
    time_repr = "{:%Y_%m_%d_%H_%M_%S_%Z}".format(timestamp)

    # Processing

    # CASAcore UVW is conjugated, so to make things consistent with data
    # streaming off telescope we need the vis flipped about
    if ARGS.ms:
        disko.vis_arr = disko.vis_arr.conjugate()
    elif ARGS.file:
        disko.vis_arr = disko.vis_arr.conjugate()
    else:
        pass

    if ARGS.show_sources:
        src_list = get_source_list(source_json, el_limit=ARGS.elevation, jy_limit=1e4)

    if ARGS.lasso:
        logger.info("L1 regularization alpha=%f" %ARGS.alpha)
        sky = disko.image_lasso(disko.vis_arr, sphere, alpha=ARGS.alpha, l1_ratio=ARGS.l1_ratio, scale=False, use_cv=ARGS.cv)
    elif ARGS.matrix_free:
        logger.info("Matrix Free alpha={}".format(ARGS.alpha))
        data = disko.vis_to_data()
        sky = disko.solve_matrix_free(data, sphere, alpha=ARGS.alpha, scale=False, lsqr=ARGS.lsqr, fista=ARGS.fista, lsmr=ARGS.lsmr, niter=ARGS.niter)
    elif ARGS.tikhonov:
        logger.info("L2 regularization alpha={}".format(ARGS.alpha))
        sky = disko.image_tikhonov(disko.vis_arr, sphere, alpha=ARGS.alpha, scale=False, usedask=ARGS.dask)
       
        if ARGS.mesh:
            for i in range(ARGS.adaptive):
                sphere.write_mesh(f"{ARGS.title}_round_{i}.vtk")

                sphere.refine()
                sky = disko.image_tikhonov(disko.vis_arr, sphere, alpha=ARGS.alpha, scale=False, usedask=ARGS.dask)
                sphere.pixels = sphere.pixels / sphere.pixel_areas

    else:
        sky = disko.solve_vis(disko.vis_arr, sphere)

    if ARGS.HDF:
        fpath = os.path.join(ARGS.dir, ARGS.HDF)
        sphere.to_hdf(fpath)

    image_title = f"{ARGS.title}_{time_repr}"

    def path(ending, image_title):
        os.makedirs(ARGS.dir, exist_ok=True)
        fname = '{}.{}'.format(image_title, ending)
        return os.path.join(ARGS.dir, fname)

    if ARGS.mesh:
        # Save as a VTK file
        sphere.write_mesh(path('vtk', image_title))


    def save_images(image_title, source_list):
        
        if ARGS.VTK:
            sphere.write_mesh(path('vtk', image_title))

        if ARGS.FITS:
            # Save as a FITS file
            sphere.to_fits(fname=path('fits', image_title), info=disko.info)
        
        if ARGS.SVG:
            fname = path('svg', image_title)
            sphere.to_svg(fname=fname, show_grid=True, src_list=source_list, title=image_title)
            logger.info("Generating {}".format(fname))
        if ARGS.PNG:
            fname = path('png', image_title)
            sphere.plot(plt, source_list)
            plt.title(image_title)
            plt.tight_layout()
            plt.savefig(fname, dpi=300)
            plt.close()
            logger.info("Generating {}".format(fname))
        if ARGS.PDF:
            fname = path('pdf', image_title)
            sphere.plot(plt, source_list)
            plt.title(image_title)
            plt.savefig(fname, dpi=600)
            plt.close()
            logger.info("Generating {}".format(fname))
        if ARGS.display:
            sphere.plot(plt, src_list)
            plt.title(image_title)
            plt.show()

    if ARGS.FITS or ARGS.SVG or ARGS.PNG or ARGS.PDF:
        save_images('{}_{}'.format(ARGS.title, time_repr), source_list=src_list)
    
    #if ARGS.SVG:
        #fname = '{}.svg'.format(image_title)
        #fpath = os.path.join(ARGS.dir, fname)

        ##sky = disko.image_lasso(disko.vis_arr, sphere, alpha=0.02, scale=False)
        #sphere.to_svg(fname=fpath, show_grid=True, src_list=src_list, fov=ARGS.fov, title=image_title)
        #logger.info("Generating {}".format(fname))
    #if ARGS.PNG:
        #sphere.plot(plt, src_list)
        #plt.title(image_title)
        #fname = '{}.png'.format(image_title)
        #fpath = os.path.join(ARGS.dir, fname)
        #plt.savefig(fpath, dpi=300)
        #plt.close()
        #logger.info("Generating {}".format(fname))
    #if ARGS.PDF:
        #sphere.plot(plt, src_list)
        #plt.title(image_title)
        #fname = '{}.pdf'.format(image_title)
        #fpath = os.path.join(ARGS.dir, fname)
        #plt.savefig(fpath, dpi=600)
        #plt.close()
        #logger.info("Generating {}".format(fname))
        
    #client.close()
