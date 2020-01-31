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
        res_min = np.radians(res_arcmin/60)
        res_max = np.radians(res_arcmax/60)
        
        # An edge_size function
        sources = 1.5*radius*(np.random.random((5,2))-0.5)+[theta, phi]
        #p1 = dmsh.Path(sources)
        
        k = 30
        x0 = 0.1 # 2.0 / k
        x = np.linspace(0,1,100)
        y = logistic(x, 1.0, k, x0)
        plt.plot(x,y,'.')
        plt.show()

        def edge_size(x):
            d = np.array([np.linalg.norm(x.T - s, axis=1) for s in sources])
            dmin = np.min(d.T, axis=1) # Distance to closest source (in diameters)
            dmin = dmin/np.max(dmin) # Normalized to 0..1
            return res_min + (res_max - res_min) * logistic(dmin, L=1.0, k=k, x0=x0)

        geo = dmsh.Circle([theta, phi], radius)

        logger.info("Generating Mesh")
        X, cells = dmsh.generate(geo, edge_size, tol=res_min/20)
        logger.info(" Mesh generated {}".format(cells.shape))

        # optionally optimize the mesh
        #logger.info("Optimizing Mesh")
        #X, cells = optimesh.odt.fixed_point_uniform(X, cells, res_min/1000, 100, verbose=True)

        # X is a list of points, cells a list of the vertices of each cell.
        logger.info("Computing Centroids Mesh")
        # Compute the centroids of the cells. These will be the pixel values.
        pixels = np.array([centroid(cell=c, points=X) for c in cells])
        

        ret.pixels = pixels
        logger.info("Computing areas Mesh")
        ret.pixel_areas = np.array([area(cell=c, points=X) for c in cells])
        
        ret.npix = ret.pixels.shape[0]
        
        logger.info("AdaptiveMeshSphere from_res, npix={}".format(ret.npix))

        x = pixels[:,0]
        y = pixels[:, 1]
        
        
        theta = np.arcsin(np.sqrt(x*x + y*y))
        phi = np.arctan2(x,y)
        
        
        ret.fov = np.degrees(radius*2)
        ret.pixels = np.zeros(ret.npix)
        
        el_r, az_r = hp2elaz(theta, phi)

        ret.el_r = el_r
        ret.az_r = az_r

        ret.l, ret.m, ret.n = elaz2lmn(ret.el_r, ret.az_r)
        if True:
            plt.plot(ret.l, ret.m, 'x')
            #plt.plot(el_r, az_r, 'x')
            plt.show()

            # and write it to a file
            meshio.write_points_cells("circle.vtk", X, {"triangle": cells})

        return ret

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

    sph = AdaptiveMeshSphere.from_resolution(res_arcmin=10, res_arcmax=180, theta=np.radians(20.0), phi=0.0, radius=np.radians(20))
    sph.to_fits('test.fits', fov=20)
