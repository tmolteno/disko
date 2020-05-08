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
import dask.array as da

from copy import deepcopy
from scipy.optimize import minimize
from sklearn import linear_model

from tart.imaging import elaz
from tart.util import constants


from .sphere import HealpixSphere
from .ms_helper import read_ms

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
            if (i < j):
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


import scipy.sparse.linalg as spalg

class DiSkOOperator(spalg.LinearOperator):
    '''
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.LinearOperator.html
    
    A subclass must implement either one of the methods _matvec and _matmat, and the attributes/properties shape (pair of integers) and dtype (may be None). It may call the __init__ on this class to have these attributes validated. Implementing _matvec automatically implements _matmat (using a naive algorithm) and vice-versa.
    
    THe linearoperatro represents the Telescope Operator with the harmonics as row_vectors
    
    A_ij = 
    
    https://pylops.readthedocs.io/en/latest/adding.html
    '''
    
    def __init__(self, u_arr, v_arr, w_arr, data, frequencies, sphere):
        self.N = sphere.npix # Number of pixels
        self.u_arr = u_arr
        self.v_arr = v_arr
        self.w_arr = w_arr
        self.dtype=DATATYPE
        
        try:
            self.n_v, self.n_freq, self.npol = data.shape
        except:
            raise RuntimeError("Data must be of the shape [n_v, n_freq, n_pol]")
            
        self.M = self.n_v * self.n_freq
        
        self.frequencies = frequencies
        self.sphere = sphere
        self.n_arr_minus_1 = self.sphere.n - 1

        self.shape = (self.M, self.N)
        logger.info("Creating LinearOperator data={}".format(self.shape))
        
    def A(self, i, j, p2j):
        u, v, w = self.u_arr[j], self.v_arr[j], self.w_arr[j]      # the row index (one u,v,w element per vis)
        l, m, n = self.sphere.l[i], self.sphere.m[i], self.sphere.n[i] # The column index (one l,m,n element per pixel)
        return np.exp(-p2j*(u*l + v*m + w*(n-1))) * self.sphere.pixel_areas

    def Ah(self, i, j, p2j):
        np.conj(self.A(j, i, p2j))
    
    def _matvec(self, x):
        '''
            Multiply by the sky x, producing the set of measurements y
            Returns returns A * x.
        '''
        y = []
        for f in self.frequencies:
            wavelength = 2.99793e8 / f
            p2j = 2*np.pi*1.0j / wavelength

            if True:
                for u, v, w in zip(self.u_arr, self.v_arr, self.w_arr):
                    column = np.exp(-p2j*(u*self.sphere.l + v*self.sphere.m + w*self.n_arr_minus_1)) * self.sphere.pixel_areas

                    y.append(np.dot(x, column))
            else:
                for col in range(self.M):
                    p = 0.0
                    
                    for row in range(self.N):
                        _x = self.A(row, col, p2j)*x[row]
                        logger.info("{} * {} = {}".format(self.A(row, col, p2j), x[row], _x))
                        p += _x
                    y.append(p)
                    logger.info("={}".format(p))
        return np.array(y)
    
    def _rmatvec(self, x):
        r'''
            Returns A^H * v, where A^H is the conjugate transpose of A.
        '''
        ret = []
        for f in self.frequencies:
            wavelength = 2.99793e8 / f
            p2j = 2*np.pi*1.0j / wavelength
            
            # Vector version
            for l, m, n_1 in zip(self.sphere.l, self.sphere.m, self.n_arr_minus_1):
                column = np.exp(p2j*(self.u_arr*l + self.v_arr*m + self.w_arr*n_1)) * self.sphere.pixel_areas  # TODO check the pixel areas here.
                ret.append(np.dot(x, column))

        return np.array(ret)
    
    
class DiSkO(object):
    
    def __init__(self, u_arr, v_arr, w_arr):
        self.harmonics = {} # Temporary store for harmonics
        self.u_arr = u_arr
        self.v_arr = v_arr
        self.w_arr = w_arr
        self.n_v = len(self.u_arr)
        
    @classmethod
    def from_ant_pos(cls, ant_pos, wavelength):
        ## Get u, v, w from the antenna positions
        baselines, u_arr, v_arr, w_arr = get_all_uvw(ant_pos, wavelength)
        ret = cls(u_arr, v_arr, w_arr)
        ret.info = {}
        return ret

    @classmethod
    def from_ms(cls, ms, num_vis, res_arcmin, chunks=1000, channel=0):
        u_arr, v_arr, w_arr, cv_vis, hdr, tstamp = read_ms(ms, num_vis, res_arcmin, chunks, channel)
        
        ret = cls(u_arr, v_arr, w_arr)
        ret.vis_arr = np.array(cv_vis, dtype=np.complex128)
        ret.timestamp = tstamp
        ret.info = hdr

        return ret


    @classmethod
    def from_cal_vis(cls, cal_vis):

        c = cal_vis.get_config()
        ant_p = np.asarray(c.get_antenna_positions())

        # We need to get the vis array to be correct for the full set of u,v,w points (baselines), 
        # including the -u,-v, -w points.

        baselines, u_arr, v_arr, w_arr = get_all_uvw(ant_p, wavelength=constants.L1_WAVELENGTH)

        ret = cls(u_arr, v_arr, w_arr)
        ret.vis_arr = []
        for bl in baselines:
            v = cal_vis.get_visibility(bl[0], bl[1])  # Handles the conjugate bit
            ret.vis_arr.append(v)
            #logger.info("vis={}, bl={}".format(v, bl))
        ret.vis_arr = np.array(ret.vis_arr, dtype=DATATYPE)
        ret.info = {}
        return ret

    def get_harmonics(self, in_sphere):
        ''' Create the harmonics for this arrangement of sphere pixels
        '''
        #cache_key = "{}:".format(in_sphere.npix)
        #if (cache_key in self.harmonics):
            #return self.harmonics[cache_key]

        n_arr_minus_1 = in_sphere.n - 1
        harmonic_list = []
        p2j = 2*np.pi*1.0j
        
        #logger.info("pixel areas:  {}".format(in_sphere.pixel_areas))
        for u, v, w in zip(self.u_arr, self.v_arr, self.w_arr):
            harmonic = np.exp(p2j*(u*in_sphere.l + v*in_sphere.m + w*n_arr_minus_1)) * in_sphere.pixel_areas
            assert(harmonic.shape[0] == in_sphere.npix)
            harmonic_list.append(harmonic)
        #self.harmonics[cache_key] = harmonic_list

        #assert(harmonic_list[0].shape[0] == in_sphere.npix)
        return harmonic_list

    def image_visibilities(self, vis_arr, sphere):
        """
        Create a DiSkO image from visibilities using the direct ajoint of the
        measurement operator (corresponds to the inverse DFT)

        Args:

            vis_arr (np.array): An array of visibilities
            sphere (int):       he healpix sphere.
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
        
        logger.info("Elapsed {}s".format(time.time() - t0))

        sphere.set_visible_pixels(sky, scale)
        
        return sky.reshape(-1,1)


    def solve_matrix_free(self, data, sphere, alpha=0.0, scale=True):
        '''
            data = [vis_arr, n_freq, n_pol]
        '''
        logger.info("Solving Visabilities nside={}".format(sphere.nside))
        t0 = time.time()

        frequencies = [1.57542e9]
        wavelength = 2.99793e8 / frequencies[0]

        A = DiSkOOperator(self.u_arr * wavelength, self.v_arr * wavelength, self.w_arr * wavelength, data, frequencies, sphere)
        if True:
            sky, lstop, itn, r1norm, r2norm, anorm, acond, arnorm, xnorm, var = spalg.lsqr(A, data, damp=alpha)
            logger.info("Matrix free solve elapsed={} x={}, stop={}, itn={} r1norm={}".format(time.time() - t0, sky.shape, lstop, itn, r1norm))      
        else:
            sky, lstop, itn, normr, mormar, morma, conda, normx = spalg.lsmr(A, data, damp=alpha)
            logger.info("Matrix free solve elapsed={} x={}, stop={}, itn={} normr={}".format(time.time() - t0, sky.shape, lstop, itn, normr))      
        #sky = np.abs(sky)
        sphere.set_visible_pixels(sky, scale)
        return sky.reshape(-1,1)

    def make_gamma(self, sphere):

        logger.info("Making Gamma Matrix npix={}".format(sphere.npix))
        t0 = time.time()

        harmonic_list = self.get_harmonics(sphere)

        n_s = len(harmonic_list[0])
        n_v = len(harmonic_list)

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

    #def sub_image(self, vis_arr, sphere, alpha, scale=True, n=4):
        #'''
            #Split an image up and image the bits separately
        #'''
        #subspheres = sphere.split(n)
        #full_soln = np.zeros_like(sphere.pixels)
        
        #results = []
        #for sph in subspheres:
            #gamma = self.make_gamma(sph)
            #proj_operator_real = np.real(gamma)
            #proj_operator_imag = np.imag(gamma)
            #proj_operator = np.block([[proj_operator_real], [proj_operator_imag]])
            
            #vis_aux = np.concatenate((np.real(vis_arr), np.imag(vis_arr)))
            
            #n_s = sph.pixels.shape[0]

            #if True:
                #reg = linear_model.ElasticNet(alpha=alpha, l1_ratio=0.0, max_iter=10000, positive=True)
                #reg.fit(proj_operator, vis_aux)
                #subsky = reg.coef_
            #else:
                #subsky = dask.linalg.linear.lstsqr()
            #results.append(subsky)
            #logger.info("subsky = {}".format(subsky.shape))
        #full_soln = np.stack(results)
            
        #sphere.set_visible_pixels(full_soln, scale)
        #return full_soln.reshape(-1,1)
        
    def image_tikhonov(self, vis_arr, sphere, alpha, scale=True, usedask=False):
        n_s = sphere.pixels.shape[0]
        n_v = self.u_arr.shape[0]
        
        lambduh = alpha/np.sqrt(n_s)
        if not usedask:
            gamma = self.make_gamma(sphere)
            logger.info("Building Augmented Operator...")
            proj_operator_real = np.real(gamma).astype(np.float32)
            proj_operator_imag = np.imag(gamma).astype(np.float32)
            gamma = None
            proj_operator = np.block([[proj_operator_real], [proj_operator_imag]])
            proj_operator_real = None
            proj_operator_imag = None 
            logger.info('augmented: {}'.format(proj_operator.shape))
            
            vis_aux = np.array(np.concatenate((np.real(vis_arr), np.imag(vis_arr))), dtype=np.float32)
            logger.info('vis mean: {} shape: {}'.format(np.mean(vis_aux), vis_aux.shape))

            logger.info("Solving...")
            reg = linear_model.ElasticNet(alpha=lambduh, l1_ratio=0.05, max_iter=10000, positive=True)
            reg.fit(proj_operator, vis_aux)
            sky = reg.coef_
            
            score = reg.score(proj_operator, vis_aux)
            logger.info('Loss function: {}'.format(score))
            
        else:
            from dask_ml.linear_model import LinearRegression
            import dask_glm
            import dask.array as da
            from dask.distributed import Client, LocalCluster
            from dask.diagnostics import ProgressBar
            import dask
            
            logger.info('Starting Dask Client')
            
            if True:
                cluster = LocalCluster(dashboard_address=':8231', processes=False)
                client = Client(cluster)
            else:
                client = Client('tcp://localhost:8786')
                
            logger.info("Client = {}".format(client))
            
            harmonic_list = []
            p2j = 2*np.pi*1.0j
            
            dl = sphere.l
            dm = sphere.m
            dn = sphere.n
        
            n_arr_minus_1 = dn - 1

            du = self.u_arr
            dv = self.v_arr
            dw = self.w_arr
        
            for u, v, w in zip(du, dv, dw):
                harmonic = da.from_array(np.exp(p2j*(u*dl + v*dm + w*n_arr_minus_1)) / np.sqrt(sphere.npix), chunks=(n_s,))
                harminc = client.persist(harmonic)
                harmonic_list.append(harmonic)

            gamma = da.stack(harmonic_list)
            logger.info('Gamma Shape: {}'.format(gamma.shape))
            #gamma = gamma.reshape((n_v, n_s))
            gamma = gamma.conj()
            gamma = client.persist(gamma)
            
            logger.info('Gamma Shape: {}'.format(gamma.shape))
            
            logger.info("Building Augmented Operator...")
            proj_operator_real = da.real(gamma)
            proj_operator_imag = da.imag(gamma)
            proj_operator = da.block([[proj_operator_real], [proj_operator_imag]])
            
            proj_operator = client.persist(proj_operator)
            
            logger.info("Proj Operator shape {}".format(proj_operator.shape))
            vis_aux = da.from_array(np.array(np.concatenate((np.real(vis_arr), np.imag(vis_arr))), dtype=np.float32))
            
            #logger.info("Solving...")

            
            en = dask_glm.regularizers.ElasticNet(weight=0.01)
            en =  dask_glm.regularizers.L2()
            #dT = da.from_array(proj_operator, chunks=(-1, 'auto'))
            ##dT = da.from_array(proj_operator, chunks=(-1, 'auto'))
            #dv = da.from_array(vis_aux)
            

            dask.config.set({'array.chunk-size': '1024MiB'})
            A = da.rechunk(proj_operator, chunks=('auto', n_s))
            A = client.persist(A)
            y = vis_aux # da.rechunk(vis_aux, chunks=('auto', n_s))
            y = client.persist(y)
            #sky = dask_glm.algorithms.proximal_grad(A, y, regularizer=en, lambduh=alpha, max_iter=10000)

            logger.info("Rechunking completed.. A= {}.".format(A.shape))
            reg =  LinearRegression(penalty=en, C=1.0/lambduh,  
                                    fit_intercept=False, 
                                    solver='lbfgs', 
                                    max_iter=1000, tol=1e-8 )
            sky = reg.fit(A, y)
            sky = reg.coef_
            score = reg.score(proj_operator, vis_aux)
            logger.info('Loss function: {}'.format(score.compute()))

        logger.info("Solving Complete: sky = {}".format(sky.shape))

        sphere.set_visible_pixels(sky, scale=True)
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
