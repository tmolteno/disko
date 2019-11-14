#!/usr/bin/env python
#
# The DiSkO algorithm for imaging without gridding.
#
# Tim Molteno 2017-2019 tim@elec.ac.nz
#
import os
import argparse
import sys
import threading
import datetime
import json
import logging
import time

import numpy as np
import healpy as hp

from copy import deepcopy
from scipy.optimize import minimize
from sklearn import linear_model

from tart.imaging import elaz
from tart.util import constants

from .sphere import HealpixSphere

'''
    Little helper function to get the UVW positions from the antennas positions.
    The test (i != j) can be changed to (i > j) to avoid the duplicated conjugate
    measurements.
'''
def get_all_uvw(ant_pos, wavelength):
    baselines = []
    num_ant = len(ant_pos)
    ant_p = np.array(ant_pos)
    for i in range(num_ant):
        for j in range(num_ant):
            if (i != j):
                baselines.append([i,j])
                
    bl_pos = ant_p[np.array(baselines).astype(int)]
    uu_a, vv_a, ww_a = (bl_pos[:,0] - bl_pos[:,1]).T/wavelength
    return baselines, uu_a, vv_a, ww_a


def to_column(x):
    return x.reshape([-1,1])


logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler()) # Add other handlers if you're using this as a library
logger.setLevel(logging.INFO)

def get_source_list(source_json, el_limit, jy_limit):
    src_list = []
    if source_json is not None:
        src_list = elaz.from_json(source_json, el_limit=el_limit, jy_limit=jy_limit)
    return src_list

DATATYPE=np.complex128

class DiSkO(object):
        
    def __init__(self, ant_pos, wavelength):
        ## Get u, v, w from the antenna positions
        self.baselines, self.u_arr, self.v_arr, self.w_arr = get_all_uvw(ant_pos, wavelength)
        self.harmonics = {} # Temporary store for harmonics
        self.n_v = len(self.u_arr)

    @classmethod
    def from_cal_vis(cls, cal_vis):

        c = cal_vis.get_config()
        ant_p = np.asarray(c.get_antenna_positions())

        # We need to get the vis array to be correct for the full set of u,v,w points (baselines), 
        # including the -u,-v, -w points.


        ret = cls(ant_p, wavelength=constants.L1_WAVELENGTH)
        ret.vis_arr = []
        for bl in ret.baselines:
            v = cal_vis.get_visibility(bl[0], bl[1])  # Handles the conjugate bit
            ret.vis_arr.append(v)
            logger.info("vis={}, bl={}".format(v, bl))
        ret.vis_arr = np.array(ret.vis_arr, dtype=DATATYPE)
        return ret
                  
    def get_harmonics(self, in_sphere):
        ''' Create the harmonics for this arrangement of sphere pixels
        '''
        cache_key = "{}:{}".format(in_sphere.nside, in_sphere.npix)
        if (cache_key in self.harmonics):
            return self.harmonics[cache_key]

        n_arr_minus_1 = in_sphere.n - 1
        harmonic_list = []
        p2j = 2*np.pi*1.0j
        
        for u, v, w in zip(self.u_arr, self.v_arr, self.w_arr):
            harmonic = np.exp(p2j*(u*in_sphere.l + v*in_sphere.m + w*n_arr_minus_1)) / np.sqrt(in_sphere.npix)
            assert(harmonic.shape[0] == in_sphere.npix)
            harmonic_list.append(harmonic)
        self.harmonics[cache_key] = harmonic_list

        assert(harmonic_list[0].shape[0] == in_sphere.npix)
        return harmonic_list

    def image_visibilities(self, vis_arr, sphere):
        """Create a DiSkO image from visibilities

        Args:

            vis_arr (np.array): An array of visibilities
            sphere (int):        The healpix sphere parameter.
        """

        assert len(vis_arr) == len(self.u_arr)
        logger.info("Imaging Visabilities nside={}".format(sphere.nside))
        t0 = time.time()
        
        pixels = np.zeros(sphere.npix, dtype=DATATYPE)
        harmonic_list = self.get_harmonics(sphere)
        for h, vis in zip(harmonic_list, vis_arr):
            pixels += vis*h
        
        t1 = time.time()
        logger.info("Elapsed {}s".format(time.time() - t0))

        sphere.set_visible_pixels(pixels)
        
        return pixels.reshape(-1,1)

    def solve_vis(self, vis_arr, sphere, scale=True):
    
        logger.info("Solving Visabilities nside={}".format(sphere.nside))
        t0 = time.time()

        gamma = self.make_gamma(sphere)
        
        sky, residuals, rank, s = np.linalg.lstsq(gamma, to_column(vis_arr), rcond=None)
        
        t1 = time.time()
        logger.info("Elapsed {}s".format(time.time() - t0))

        sphere.set_visible_pixels(sky, scale)
        
        return sky.reshape(-1,1)

    def make_gamma(self, sphere):

        logger.info("Making Gamma Matrix nside={} npix={}".format(sphere.nside, sphere.npix))
        t0 = time.time()

        harmonic_list = self.get_harmonics(sphere)

        n_s = len(harmonic_list[0])
        n_v = len(harmonic_list)
        
        print(harmonic_list[0].shape)
        assert(harmonic_list[0].shape[0] == sphere.npix)

        if False:
            gamma = np.zeros((n_v, n_s), dtype=DATATYPE)
            for i in range(n_v):
                gamma[i, :] = harmonic_list[i].reshape(n_s).conj()
        else:
            gamma = np.array(harmonic_list, dtype=DATATYPE)
            gamma = gamma.reshape((n_v, n_s))
            gamma = gamma.conj()
            
        logger.info('Gamma Shape: {}'.format(gamma.shape))
        #for i, h in enumerate(harmonic_list):
            #gamma[i,:] = h[0]
        
        return gamma

    def image_lasso(self, vis_arr, sphere, alpha, scale=True, use_cv=False):
        gamma = self.make_gamma(sphere)
        
        proj_operator_real = np.real(gamma)
        proj_operator_imag = np.imag(gamma)
        proj_operator = np.block([[proj_operator_real], [proj_operator_imag]])
        
        vis_aux = np.concatenate((np.real(vis_arr), np.imag(vis_arr)))
        
        # Save proj operator for Further Analysis.
        if False:
            fname = "l1_big_files.npz"
            np.savez_compressed(fname, gamma_re=proj_operator_real, gamma_im=proj_operator_imag, vis_re=np.real(vis_arr), vis_im=np.imag(vis_arr))
            logger.info("Operator file {} saved".format(fname))
            
            logger.info("proj_operator = {}".format(proj_operator.shape))
            logger.info("vis_aux = {}".format(vis_aux.shape))
        
        n_s = sphere.pixels.shape[0]
        
        if not use_cv:
            reg = linear_model.ElasticNet(alpha=alpha/np.sqrt(n_s), l1_ratio=1.0, max_iter=10000, positive=True)
            reg.fit(proj_operator, vis_aux)
        else:
            reg = linear_model.ElasticNetCV(l1_ratio=1.0, cv=5, max_iter=10000, positive=True)
            reg.fit(proj_operator, vis_aux)
            logger.info("Cross Validation = {}".format(reg.alpha_))

        sky = reg.coef_
        logger.info("sky = {}".format(sky.shape))

        sphere.set_visible_pixels(sky, scale)
        return sky.reshape(-1,1)

    def image_tikhonov(self, vis_arr, sphere, alpha, scale=True):
        gamma = self.make_gamma(sphere)
        
        proj_operator_real = np.real(gamma)
        proj_operator_imag = np.imag(gamma)
        proj_operator = np.block([[proj_operator_real], [proj_operator_imag]])
        
        vis_aux = np.concatenate((np.real(vis_arr), np.imag(vis_arr)))
        
        n_s = sphere.pixels.shape[0]

        reg = linear_model.ElasticNet(alpha=alpha/np.sqrt(n_s), l1_ratio=0.0, max_iter=10000, positive=True)
        reg.fit(proj_operator, vis_aux)

        sky = reg.coef_
        logger.info("sky = {}".format(sky.shape))

        sphere.set_visible_pixels(sky, scale)
        return sky.reshape(-1,1)


    @classmethod
    def plot(self, plt, sphere, src_list):
        rot = (0, 90, 0)
        plt.figure() # (figsize=(6,6))
        logger.info('sphere.pixels: {}'.format(sphere.pixels.shape))
        if True:
            hp.orthview(sphere.pixels, rot=rot, xsize=1000, cbar=True, half_sky=True, hold=True)
            hp.graticule(verbose=False)
            plt.tight_layout()
        else:
            hp.mollview(sphere.pixels, rot=rot, xsize=1000, cbar=True)
            hp.graticule(verbose=True)
        
        if src_list is not None:
            for s in src_list:
                sphere.plot_x(s.el_r, s.az_r)
        
    def display(self, plt, src_list, nside):
        sphere = HealpixSphere(nside)
        sky = self.solve_vis(self.vis_arr, sphere)
        sphere.plot(plt, src_list)

    def beam(self, plt, nside):
        sphere = HealpixSphere(nside)
        sky = self.solve_vis(np.ones_like(self.vis_arr), nside)
        sphere.plot(plt, src_list=None)
