# https://stackoverflow.com/questions/7975522/mesh-generation-for-computational-science-in-python

import logging
import dmsh
import optimesh
import meshio
import scipy

from .sphere import HealpixSphere, hp2elaz, elaz2lmn

import numpy as np
import matplotlib.pyplot as plt



logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler()) # Add other handlers if you're using this as a library
logger.setLevel(logging.INFO)


def centroid(cell, points):
    return np.sum(points[cell].T, axis=1)/3

def area(cell, points):
    p, q, r = points[cell]
    return np.abs(0.5*(p[0]*(q[1]-r[1])+q[0]*(r[1]-p[1])+r[0]*(p[1]-q[1])))
    
def logistic(x, L, k, x0):
    return L / (1.0 + np.exp(-k*(x - x0)))

class Sphere:
    def f(self, x):
        return 1.0 - (x[0] ** 2 + x[1] ** 2 + x[2] ** 2)

    def grad(self, x):
        return -2 * x

class AdaptiveMeshSphere(HealpixSphere):
    ''' 
        An adaptive mesh sphere.
    '''
    def __init__(self, resolution_arcmin):
        self.res_arcmin = resolution_arcmin
        self.npix = None
        self.pixel_areas = None

        logger.info("New AdaptiveMeshSphere, resolution_arcmin={}".format(self.res_arcmin))

    @classmethod
    def from_resolution(cls, res_arcmin=None, res_arcmax=None, theta=0.0, phi=0.0, radius=0.0):
        # Theta is co-latitude measured southward from the north pole
        # Phi is [0..2pi]
        
        ret = cls(res_arcmin)
        ret.res_min = np.radians(res_arcmin/60)
        ret.res_max = np.radians(res_arcmax/60)
        
        geo = dmsh.Circle([theta, phi], radius)

        logger.info("Generating Mesh")
        ret.X, ret.cells = dmsh.generate(geo, ret.res_max, tol=ret.res_min/100)
        logger.info(" Mesh generated {}".format(ret.cells.shape))

        #logger.info("Optimizing Mesh")
        #ret.X, ret.cells = optimesh.cvt.quasi_newton_uniform_full(ret.X, ret.cells, 1e-2, 100, verbose=True)

        ret.npix = ret.cells.shape[0]
        ret.pixels = np.zeros(ret.npix)
        ret.radius = radius
        ret.fov = np.degrees(radius*2)
        
        ret.refresh()
        
        logger.info("AdaptiveMeshSphere from_res, npix={}".format(ret.npix))

        return ret

    def refresh(self):
        self.compute_points()
        self.compute_areas()
        self.set_lmn()

    def write_mesh(self, fname='output.vtk'):
        #import matplotlib.pyplot as plt
        
        #plt.plot(self.l, self.m, 'x')
        ###plt.plot(el_r, az_r, 'x')
        #plt.show()

        # and write it to a file
        meshio.write_points_cells(fname, self.X, {"triangle": self.cells}, 
                                  cell_data={'triangle': {'flux': self.pixels}})

    def compute_points(self):
        logger.info("Computing Centroids")
        self.points = np.array([centroid(cell=c, points=self.X) for c in self.cells])

    def compute_areas(self):
        logger.info("Computing areas")
        pixel_areas = np.array([area(cell=c, points=self.X) for c in self.cells])
        total_area = np.sum(pixel_areas)
        self.pixel_areas = pixel_areas / total_area

    def refine(self):
        '''
            refine the mesh by dividing each simplex into three
            based on the values of the pixels in the existing mesh.
            
            The pixel value is flux, so each sub-element should be assigned 1/3 the value,
            and the new pixels used as a starting point for the next lsqr solution.
        '''
        p05, p50, p95 = np.percentile(self.pixels, [5, 50, 95])
        logging.info("Data {} {} {}".format(p05, p50, p95))
        
        refined_points = []
        refined_cells = []
        refined_pixels = []
        
        for y, c in zip(self.pixels, self.cells):
            p1, p2, p3 = self.X[c]

            refined_points.append(p1); p1_index = len(refined_points)-1;
            refined_points.append(p2); p2_index = len(refined_points)-1;
            refined_points.append(p3); p3_index = len(refined_points)-1;
            if (y > p95):
                # Split the cell, bisecting each edge
                #
                #    p1       p12          p2
                #
                #       p13          p23
                #
                #             p3
                p12 = (p1 + p2) / 2
                p13 = (p1 + p3) / 2
                p23 = (p2 + p3) / 2
                refined_points.append(p12); p12_index = len(refined_points)-1;
                refined_points.append(p13); p13_index = len(refined_points)-1;
                refined_points.append(p23); p23_index = len(refined_points)-1;
                
                refined_cells.append([p1_index, p12_index, p13_index])
                refined_cells.append([p12_index, p2_index, p23_index])
                refined_cells.append([p23_index, p3_index, p13_index])
                refined_cells.append([p13_index, p23_index, p12_index])
                
                refined_pixels.append(y)
                refined_pixels.append(y)
                refined_pixels.append(y)
                refined_pixels.append(y)
                
            else:
                refined_cells.append([p1_index, p2_index, p3_index])
                refined_pixels.append(y)

                
        self.X = np.array(refined_points)
        self.cells = np.array(refined_cells)
        self.pixels = np.array(refined_pixels)
        self.npix = self.cells.shape[0]
        
        self.refresh()
        
    def set_lmn(self):
        x = self.points[:, 0]
        y = self.points[:, 1]
        r = np.sqrt(x*x + y*y)
        
        # Convert the x,y to theta and phi
        
        theta = np.arcsin(r)
        phi = np.arctan2(x,y)
        
        el_r, az_r = hp2elaz(theta, phi)

        self.el_r = el_r
        self.az_r = az_r

        self.l, self.m, self.n = elaz2lmn(self.el_r, self.az_r)


        
if __name__=="__main__":
    
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    fh = logging.FileHandler('disko.log')
    fh.setLevel(logging.INFO)
    
    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # add formatter to ch
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)

    # add ch to logger
    logger.addHandler(ch)
    logger.addHandler(fh)

    sph = AdaptiveMeshSphere.from_resolution(res_arcmin=10, res_arcmax=180, theta=np.radians(00.0), phi=0.0, radius=np.radians(20))
    sph.pixels = np.random.random(sph.npix)
    sph.refine()
    sph.write_mesh()
    sph.to_fits('test.fits', fov=20)
