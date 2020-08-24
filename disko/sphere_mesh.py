# https://stackoverflow.com/questions/7975522/mesh-generation-for-computational-science-in-python

import logging
import dmsh
#import optimesh
import scipy

from scipy.spatial import Delaunay, delaunay_plot_2d
import meshio

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
    
#def logistic(x, L, k, x0):
    #return L / (1.0 + np.exp(-k*(x - x0)))

#class Sphere:
    #def f(self, x):
        #return 1.0 - (x[0] ** 2 + x[1] ** 2 + x[2] ** 2)

    #def grad(self, x):
        #return -2 * x

class AdaptiveMeshSphere(HealpixSphere):
    ''' 
        An adaptive mesh sphere.
    '''
    def __init__(self, res_min, res_max, radius):
        self.radius = radius
        self.fov = np.degrees(radius*2)
        
        self.res_max = res_max
        self.res_min = res_min
        self.res_arcmin = np.degrees(res_min*60)
        
        geo = dmsh.Circle([0.0, 0.0], 1.0)

        N = 2*np.pi*radius*radius / (res_max*res_max)
        
        logger.info("Generating Mesh: r: {}, res: {},  N = {}".format(radius, (res_min, res_max), N))
        X, cells = dmsh.generate(geo, res_max/radius, tol=res_min/100)
        logger.info(" Mesh generated {}".format(cells.shape))
        
        #logger.info("Optimizing Mesh")
        #X, cells = optimesh.odt.fixed_point_uniform(X, cells, 1e-2, 10, verbose=True)
        
        self.mesh(X)

        logger.info("New AdaptiveMeshSphere, resolution_min={}".format(self.res_arcmin))



    @classmethod
    def from_resolution(cls, res_arcmin=None, res_arcmax=None, theta=0.0, phi=0.0, radius=0.0):
        # Theta is co-latitude measured southward from the north pole
        # Phi is [0..2pi]
        
        res_max = np.radians(res_arcmax / 60)
        res_min = np.radians(res_arcmin / 60)
        ret = cls(res_min, res_max, radius)        
        logger.info("AdaptiveMeshSphere from_res, npix={}".format(ret.npix))

        return ret

    def mesh(self, pts):
        logger.info("Meshing {}".format(pts.shape))
        self.tri = Delaunay(pts)
        
        #logger.info("Optimizing Mesh {} {}".format(self.tri.points.shape, self.tri.simplices.shape))
        #X, cells = optimesh.cpt.linear_solve_density_preserving(self.tri.points, self.tri.simplices.copy(), 1.0e-10, 100, verbose=True)
        #self.tri = Delaunay(X)

        self.npix = self.tri.simplices.shape[0]
        logger.info("New Mesh {}".format(self.npix))
        self.pixels = np.zeros(self.npix)
        
        # Scale points
        self.points = np.sum(self.tri.points[self.tri.simplices], axis=1)/3
        pixel_areas = self.radius*self.radius*np.array([area(cell=c, points=self.tri.points) for c in self.tri.simplices])
        total_area = np.sum(pixel_areas)
        self.pixel_areas = pixel_areas / total_area

        self.set_lmn()

    def gradient(self):
        # Return a gradient between every pair of cells
        ret = []
        cell_pairs = []
        
        r_nyquist = self.res_min/self.radius
        logger.info("R limit: {}".format(r_nyquist))

        n_ignored = 0
        for p1, nlist in enumerate(self.tri.neighbors):
            y1 = self.pixels[p1]
            #print(p1, nlist)
            for p2 in nlist:
                if p2 != -1:
                    dx, dy = self.points[p2] - self.points[p1]
                    r = np.sqrt(dx*dx + dy*dy)
                    if (r > r_nyquist): 
                        grad = (y1 - self.pixels[p2])/r # TODO Check this division by /r
                        ret.append([grad, r])
                        cell_pairs.append([p1, p2])
                    else:
                        n_ignored += 1
        logger.info("Gradient Ignored: {} of {} points".format(n_ignored, self.npix))
        
        return np.array(ret), cell_pairs
    
    
    def refine(self):
        grad, pairs = self.gradient()

        self.refine_adding(grad, pairs)
        
    def refine_adding(self, gradr, pairs):
        
        logger.info("gradr {}".format(gradr.shape))
        
        grad = gradr[:,0]
        rlist = gradr[:,1]
        
        p05, p50, p95 = np.percentile(grad, [5, 50, 95])
        logger.info("Grad Percentiles: 5: {} 50: {} 95: {}".format(p05, p50, p95))
        r05, r50, r95 = np.percentile(rlist, [5, 50, 95])
        logger.info("r Percentiles: 5: {} 50: {} 95: {}".format(r05, r50, r95))

        new_pts = self.refine_removing()
        
        new_count = 0
        for g, r, p in zip(grad, rlist, pairs):
            if g > p95:
                pt = (self.points[p[0]] + self.points[p[1]])/ 2
                new_pts.append(pt)
                new_count += 1
    
        logger.info("Added {} points to {} -> ".format(new_count, self.tri.points.shape))

        #self.tri.add_points(new_pts)
        #self.optimize()
        self.mesh(np.array(new_pts))
        
    def refine_removing(self):
        # https://stackoverflow.com/questions/35298360/remove-simplex-from-scipy-delaunay-triangulation
        if self.npix < 10000:
            return [p for p in self.tri.points] 

        # The average gradient of each cell
        gradlist = []
        edgelist = []
        for p1, nlist in enumerate(self.tri.neighbors):
            y1 = self.pixels[p1]
            #print(p1, nlist)
            g = 0
            edge = False
            for p2 in nlist:
                if p2 != -1:
                    dx, dy = self.points[p2] - self.points[p1]
                    r = np.sqrt(dx*dx + dy*dy)
                    grad = (y1 - self.pixels[p2])/r # TODO Fix Gradient /r
                    g += (grad*grad)
                else:
                    edge = True
            g = np.sqrt(g) / len(nlist)   # average gradient
            gradlist.append(g)
            edgelist.append(edge)
        
        grad = np.abs(np.array(gradlist))
        p05, p50, p95 = np.percentile(grad, [5, 50, 95])
        logger.info("Grad Percentiles: 5: {} 50: {} 95: {}".format(p05, p50, p95))
        
        new_indices = [] # Just store the indices
        # Now remove entire cells
        for p, g, e in zip(self.tri.simplices, grad, edgelist):
            if e or (g > p05):
                new_indices += list(p)  ### ERROR FIXME this adds points multiple times! Only add points sensibly!
                    
        new_indices = np.unique(new_indices)
        logger.info("Removed: {} points to {} -> ".format(self.tri.points.shape, new_indices.shape))

        return list(self.tri.points[new_indices].copy())


    def plot(self):
        import matplotlib.pyplot as plt
        #plt.plot(pts[:,0], pts[:,1], '.')
        plt.clf()
        plt.plot(self.tri.points[:,0], self.tri.points[:,1], 'o')
        plt.plot(self.points[:,0], self.points[:,1], '.')
        plt.triplot(self.tri.points[:,0], self.tri.points[:,1], self.tri.simplices.copy())
        plt.show()

        
    def write_mesh(self, fname='output.vtk'):
        #import matplotlib.pyplot as plt
        
        #plt.plot(self.l, self.m, 'x')
        ###plt.plot(el_r, az_r, 'x')
        #plt.show()

        # and write it to a file
        meshio.write_points_cells(fname, self.tri.points, [("triangle", self.tri.simplices)], 
                                  cell_data={'flux': [self.pixels]})

        
    def set_lmn(self):
        x = self.points[:, 0]*self.radius
        y = self.points[:, 1]*self.radius
        r = np.sqrt(x*x + y*y)
        
        # Convert the x,y to theta and phi
        
        theta = np.arcsin(r)
        phi = np.arctan2(x,y)
        
        el_r, az_r = hp2elaz(theta, phi)

        self.el_r = el_r
        self.az_r = az_r

        self.l, self.m, n = elaz2lmn(self.el_r, self.az_r)
        self.n_minus_1 = n - 1


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

    sph = AdaptiveMeshSphere.from_resolution(res_arcmin=10, res_arcmax=180, theta=np.radians(0.0), phi=0.0, radius=np.radians(20))
    sph.pixels = np.random.random(sph.npix)
    sph.plot()
    sph.refine()
    sph.plot()
    sph.write_mesh()
    sph.to_fits('test.fits', fov=20)
