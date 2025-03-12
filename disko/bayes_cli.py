#!/usr/bin/env python
import matplotlib
import os

import matplotlib.pyplot as plt

import argparse
import datetime
import json
import logging
import time

from copy import deepcopy

import numpy as np
import scipy.special

from tart.operation import settings

from tart_tools import api_imaging
from tart.imaging import elaz
from tart.imaging import visibility
from tart.imaging import calibration

from .disko import DiSkO
from .telescope_operator import TelescopeOperator

from .multivariate_gaussian import MultivariateGaussian

from .parser_support import sphere_from_args, sphere_args_parser

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler()) # Add other handlers if you're using this as a library
logger.setLevel(logging.INFO)


def create_prior(vis_arr, sphere, hdf_prior):
    ''' Based on the size of the visibilities, try and calculate
        what range the image should have.
    '''
    if hdf_prior is not None:
        return MultivariateGaussian.from_hdf5(hdf_prior)

    vabs = np.abs(vis_arr)

    p05, p50, p95, p100 = np.percentile(vabs, [5, 50, 95, 100])

    var = p95*p95
    logger.info("Extimated Sky Prior variance={}".format(var))
    prior = MultivariateGaussian(np.zeros(sphere.npix) + p50, sigma=p95*np.identity(sphere.npix))

    return prior


def do_inference(disko, sphere, prior, sigma_v=None):
    real_vis = vis_to_real(disko.vis_arr)

    to = TelescopeOperator(disko, sphere)

    # Transform to the natural basis.
    n_prior =  prior.linear_transform(to.Vh)
    n_v = real_vis.shape[0]

    # TODO create a proper covariance that ensures the real and imaginary components are linked.
    if sigma_v is None:
        diag = np.diag(disko.rms**2)
    else:
        diag = np.diag(np.ones(n_v // 2)*(sigma_v)**2)

    logger.info(f"do_inference(sigma_v={diag[0,0]})")

    sigma_vis = np.block([[diag, 0.5*diag],[0.5*diag, diag]]) # .rechunk('auto')

    # now invert sigma_vis
    sigma_precision = MultivariateGaussian.sp_inv(sigma_vis)
    del sigma_vis

    if True:
        prior_r = n_prior.block(0,to.rank)
        prior_n = n_prior.block(to.rank,to.n_s)

        A_r = to.A_r
        V = to.V

        del to
        posterior_r = prior_r.bayes_update(sigma_precision, real_vis, A_r)
        posterior_n = prior_n

        del A_r
        del sigma_precision
        del prior_r
        del prior_n
        del n_prior

        posterior = MultivariateGaussian.outer(posterior_r, posterior_n)

        del posterior_r
        del posterior_n

        logger.info("Transforming posterior")

        posterior = posterior.linear_transform(V)
                
        del V
    else:
        posterior = to.sequential_inference(n_prior, real_vis, sigma_precision)
        del to
        del sigma_precision
        del n_prior

    return posterior


def handle_bayes(ARGS):

    sphere = sphere_from_args(ARGS)

    # Create a prior.
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

        if ARGS.sigma_v is None:
            raise RuntimeError("The --sigma-v option must be supplied when --file JSON input is used")

        prior = create_prior(cv.v, sphere, ARGS.prior)
        timestamp = cv.get_timestamp()
        disko = DiSkO.from_cal_vis(cv)

        posterior = do_inference(disko, sphere, prior, sigma_v=ARGS.sigma_v)
        handle_output(ARGS, timestamp, posterior, sphere)

    elif ARGS.hdf:
        logger.info(f"Getting data from file {ARGS.hdf}")
        if ARGS.sigma_v is None:
            raise RuntimeError("The --sigma-v option must be supplied when HDF5 input is used")
        
        data = visibility.from_hdf5(ARGS.hdf)
        
        prior = create_prior(data['vis_list'][0].v, sphere, ARGS.prior)
        posterior = None
        
        for v in data['vis_list']:
            if posterior is not None:
                prior = posterior
            cv = calibration.CalibratedVisibility(v)
            cv.set_config(v.config)
            cv.set_phase_offset(list(range(cv.get_config().get_num_antenna())),np.array(data['phase_offset']))
            cv.set_gain(list(range(cv.get_config().get_num_antenna())),np.array(data['gain']))
            timestamp = cv.get_timestamp()
            disko = DiSkO.from_cal_vis(cv)

            # TODO Calibrate the vis with gains and phases?
            posterior = do_inference(disko, sphere, prior, sigma_v=ARGS.sigma_v)
            handle_output(ARGS, timestamp, posterior, sphere)
    else:
        logger.info("Getting Data from MS file: {}".format(ARGS.ms))

        disko = DiSkO.from_ms(ARGS.ms, ARGS.nvis, res_arcmin=sphere.res_arcmin, channel=ARGS.channel, field_id=ARGS.field)
        # Convert from reduced Julian Date to timestamp.
        timestamp = disko.timestamp
        src_list = None
        
        prior = create_prior(disko.vis_arr, sphere, ARGS.prior)
            
        posterior = do_inference(disko, sphere, prior, sigma_v=ARGS.sigma_v)
        handle_output(ARGS, timestamp, posterior, sphere)
        

def handle_output(ARGS, timestamp, posterior, sphere):

    if not ARGS.show_sources:
        src_list = None

    time_repr = "{:%Y_%m_%d_%H_%M_%S_%Z}".format(timestamp)

    # Now save the files.
    if ARGS.posterior is not None:
        posterior.to_hdf5(ARGS.posterior)

    def path(ending, image_title):
        os.makedirs(ARGS.dir, exist_ok=True)
        fname = '{}.{}'.format(image_title, ending)
        return os.path.join(ARGS.dir, fname)

    def save_images(image_title, source_list):
        # Save as a FITS file
        global disko
        
        if ARGS.FITS:
            sphere.to_fits(fname=path('fits', image_title), fov=ARGS.fov, info=disko.info)
        
        if ARGS.SVG:
            fname = path('svg', image_title)
            sphere.to_svg(fname=fname, show_grid=True, src_list=source_list, fov=ARGS.fov, title=image_title, show_cbar=True)
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

    if ARGS.PDF or ARGS.PNG or ARGS.SVG or ARGS.FITS: 


        if ARGS.mu:
            logger.info("Computing pixels")
            tic = time.perf_counter()    
            #mu_positive = np.array(da.clip(posterior.mu, 0, None))
            logger.info(f"    Took {time.perf_counter() - tic:0.4f} seconds")
            stat = sphere.set_visible_pixels(np.array(posterior.mu), scale=False)
            stat['sigma-v'] = ARGS.sigma_v
            logger.info(json.dumps(stat, sort_keys=True))
            save_images('{}_{}_mu'.format(ARGS.title, time_repr), source_list=src_list)

        if ARGS.var:
            tic = time.perf_counter()    
            logger.info("Computing variance...")
            variance = np.array(posterior.variance())
            logger.info(f"    Took {time.perf_counter() - tic:0.4f} seconds")
            sphere.set_visible_pixels(variance, scale=False)
            save_images('{}_{}_var'.format(ARGS.title, time_repr), source_list=None)
        
        if ARGS.pcf:
            tic = time.perf_counter()    
            logger.info("Computing point covariance...")
            
            brightest_pixel = np.argmax(posterior.mu)
            pix_cov=np.array(posterior.sigma()[brightest_pixel,:])
            logger.info(f"    Took {time.perf_counter() - tic:0.4f} seconds")

            sphere.set_visible_pixels(pix_cov, scale=False)
            save_images('{}_{}_pcf'.format(ARGS.title, time_repr), source_list=None)

        for i in range(ARGS.nsamples):
            sphere.set_visible_pixels(posterior.sample(), scale=False)
            save_images(image_title = '{}_{}_s{:0>5}'.format(ARGS.title, time_repr, i), source_list=None)


def main():
    np.random.seed(42)
    sphere_parsers = sphere_args_parser()

    parser = argparse.ArgumentParser(
        description='DiSkO: Bayesian inference of a posterior sky',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=sphere_parsers)


    parser.add_argument('--hdf', required=False, default=None, help="Exported Multi-visibility file")
    parser.add_argument('--ms', required=False, default=None, help="visibility file")
    parser.add_argument('--file', required=False, default=None, help="Snapshot observation saved JSON file (visiblities, positions and more).")
    
    
    parser.add_argument('--channel', type=int, default=0, help="Use this frequency channel.")
    parser.add_argument('--field', type=int, default=0, help="Use this FIELD_ID from the measurement set.")

    parser.add_argument('--dir', required=False, default='.', help="Output directory.")
    parser.add_argument('--nvis', type=int, default=1000, help="Number of visibilities to use.")
    parser.add_argument('--arcmin', type=float, default=None, help="Highest allowed res of the sky in arc minutes.")

    parser.add_argument('--sigma-v', type=float, default=None, help="Diagonal components of the visibility covariance. If not supplied use measurement set values")

    parser.add_argument('--PNG', action="store_true", help="Generate a PNG format image.")
    parser.add_argument('--PDF', action="store_true", help="Generate a PDF format image.")
    parser.add_argument('--SVG', action="store_true", help="Generate a SVG format image.")
    parser.add_argument('--FITS', action="store_true", help="Generate a FITS format image.")
    parser.add_argument('--show-sources', action="store_true", help="Show known sources on images (only works on PNG & SVG).")

    parser.add_argument('--prior', type=str, default=None, help="Load the from an HDF5 file.")
    parser.add_argument('--posterior', type=str, default=None, help="Store the posterior in HDF5 format file.")

    parser.add_argument('--uv', action="store_true", help="Plot the UV coverage.")
    parser.add_argument('--mu', action="store_true", help="Save the mean image.")
    parser.add_argument('--pcf', action="store_true", help="Save the point covariance function image.")
    parser.add_argument('--var', action="store_true", help="Save the pixel variance image.")
    parser.add_argument('--nsamples', type=int, default=0, help="Number of samples to save from the posterior.")

    parser.add_argument('--title', required=False, default="disko", help="Prefix the output files.")

    source_json = None


    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(format=log_fmt, level=logging.INFO)

    root = logging.getLogger()
    
    fh = logging.FileHandler('disko.log')
    #fh.setLevel(logging.INFO)
    
    # create console handler and set level to debug
    ch = logging.StreamHandler()
    #ch.setLevel(logging.INFO)

    # create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # add formatter to ch
    fh.setFormatter(formatter)

    # add ch to logger
    #root.addHandler(ch)
    root.addHandler(fh)

    #client = Client()

    handle_bayes(parser.parse_args())

    #client.close()
    #local_cluster.close()
