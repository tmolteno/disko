# Classes to hold an unstructured mesh sphere
# Tim Molteno tim@elec.ac.nz 2019-2022
#


# https://stackoverflow.com/questions/7975522/mesh-generation-for-computational-science-in-python

import logging
import dmsh

import optimesh

from scipy.spatial import Delaunay
import meshio

from .sphere import Sphere, hp2elaz, elaz2lmn
from .resolution import Resolution

import numpy as np

logger = logging.getLogger(__name__)
logger.addHandler(
    logging.NullHandler()
)  # Add other handlers if you're using this as a library
logger.setLevel(logging.INFO)


def centroid(cell, points):
    return np.sum(points[cell].T, axis=1) / 3


def area(cell, points):
    p, q, r = points[cell]
    return np.abs(
        0.5 * (p[0] * (q[1] - r[1]) + q[0] *
               (r[1] - p[1]) + r[0] * (p[1] - q[1]))
    )


# def logistic(x, L, k, x0):
# return L / (1.0 + np.exp(-k*(x - x0)))

# class Sphere:
# def f(self, x):
# return 1.0 - (x[0] ** 2 + x[1] ** 2 + x[2] ** 2)

# def grad(self, x):
# return -2 * x

# import pygmsh

# class AdaptiveMeshSphereNew(Sphere):

    # def __init__(self, res_min, res_max, radius_rad):
    # self.radius_rad = radius_rad
    # self.fov = np.degrees(radius_rad * 2)
    # self.res_arcmin = np.degrees(res_max)*60

    # self.res_max = res_max
    # self.res_min = res_min

    # edge_size = res_max / radius_rad
    # logger.info(f" Starting mesh generation {pygmsh.__version__}")

    # with pygmsh.geo.Geometry() as geom:
    # geom.add_circle(
    # [0.0, 0.0, 0.0],
    # 1.0,
    # mesh_size=edge_size,
    # num_sections=4,
    # compound=True,
    # )
    # mesh = geom.generate_mesh()
    # print(mesh)
    # X = mesh.points
    # cells = mesh.get_cells_type("triangle")
    # logger.info(f" Mesh generated: pts={X.shape}, cells={cells.shape}")
    # print(cells)

    # @classmethod
    # def from_resolution(
    # cls, res_arcmin=None, res_arcmax=None, theta=0.0, phi=0.0, radius_rad=0.0
    # ):
    # Theta is co-latitude measured southward from the north pole
    # Phi is [0..2pi]

    # res_max = np.radians(res_arcmax / 60)
    # res_min = np.radians(res_arcmin / 60)
    # ret = cls(res_min, res_max, radius_rad)
    # logger.info("AdaptiveMeshSphere from_res, npix={}".format(ret.npix))

    # return ret

# def get_mesh_gmsh(radius_rad, edge_size):

    # logger.info(f"Generating Mesh: Radius: {Resolution.from_rad(radius_rad)}, edge = {Resolution.from_rad(edge_size)}")
    # logger.info(f" Starting mesh generation {pygmsh.__version__}")

    # with pygmsh.geo.Geometry() as geom:
    # geom.add_circle(
    # [0.0, 0.0, 0.0],
    # 1.0,
    # mesh_size=edge_size
    # )
    # gmsh.option.setNumber("General.ExpertMode", 1)
    # mesh = geom.generate_mesh()
    # X = mesh.points
    # cells = np.array(mesh.get_cells_type("triangle"), dtype=np.int64)

    # logger.info("Optimizing Mesh")
    # X, cells = optimesh.optimize_points_cells(X, cells,  "CVT (block-diagonal)", 1e-5, 10, verbose=False)

    # return X*radius_rad, cells

def get_mesh(radius_rad, edge_size):

    logger.info(
        f"Generating Mesh: Radius: {Resolution.from_rad(radius_rad)}, edge = {Resolution.from_rad(edge_size)}")
    geo = dmsh.Circle(x0=[0.0, 0.0], r=1)
    X, cells = dmsh.generate(geo, edge_size/radius_rad,
                             tol=edge_size / 250,
                             max_steps=1000,
                             verbose=False)
    logger.info(f" Mesh generated: pts={X.shape}, cells={cells.shape}")
    logger.info("Optimizing Mesh")
    X, cells = optimesh.optimize_points_cells(
        X, cells,  "CVT (block-diagonal)", 1e-5, 10, verbose=False)

    return X*radius_rad, cells


def get_lmn(radius_rad, edge_size):
    X, cells = get_mesh(radius_rad, edge_size)

    pixel_areas = (
        np.array(
            [area(cell=c, points=X) for c in cells]
        )
    )

    centroids = np.sum(X[cells], axis=1) / 3

    x = centroids[:, 0]
    y = centroids[:, 1]
    r = np.sqrt(x * x + y * y)  # in radians

    '''
        Convert the x,y to theta and phi,
                                                        |r
                                                        |
        obs----------------------1.0---------------------  
        sin(el_r) = r

    '''

    theta = r
    phi = np.arctan2(x, y)

    el_r = np.pi / 2 - theta
    az_r = -phi

    l = np.sin(az_r) * np.cos(el_r)
    m = np.cos(az_r) * np.cos(el_r)
    # Often written in this weird way... np.sqrt(1.0 - l**2 - m**2)
    n = np.sin(el_r)

    return centroids, cells, pixel_areas, el_r, az_r, l, m, n


class AdaptiveMeshSphere(Sphere):
    """
    An adaptive mesh sphere.
    """

    def __init__(self, res_min, res_max, fov):
        logger.info(
            f"New AdaptiveMeshSphere(fov={fov}) res_min={res_min}, res_max={res_max}")
        self.radius_rad = fov.radians() / 2
        self.fov = fov
        self.res_arcmin = res_max
        self.res_max = res_max
        self.res_min = res_min

        points, simplices, pixel_areas, el_r, az_r, l, m, n = get_lmn(
            self.radius_rad, self.res_max.radians())

        self.l = l
        self.m = m
        self.n_minus_1 = n - 1

        self.el_r = el_r
        self.az_r = az_r

        self.npix = simplices.shape[0]
        self.pixels = np.zeros(self.npix)

        self.points = points

        self.simplices = simplices
        total_area = np.sum(pixel_areas)
        logger.info(f"Total area {total_area}")

        self.pixel_areas = pixel_areas / total_area

    def min_res(self):
        return self.res_min

    def __repr__(self):
        return f"AdaptiveMeshSphere fov={self.fov}, res_min={self.res_min}, N={self.npix}"

    @classmethod
    def from_resolution(cls, res_min=None, res_max=None, theta=0.0, phi=0.0, fov=None):
        # Theta is co-latitude measured southward from the north pole
        # Phi is [0..2pi]

        ret = cls(res_min, res_max, fov)
        logger.info(f"AdaptiveMeshSphere from_res, npix={ret.npix}")

        return ret

    def fast_mesh(self, pts, simplices):

        self.npix = simplices.shape[0]
        logger.info(f"Fast Mesh {self.npix}")
        self.pixels = np.zeros(self.npix)

        # Scale points
        self.points = self.radius_rad*np.sum(pts[simplices], axis=1) / 3
        self.simplices = simplices
        pixel_areas = (
            np.array(
                [area(cell=c, points=pts) for c in simplices]
            )
        )
        total_area = np.sum(pixel_areas)

        logger.info(f"Total area {total_area}")

        self.pixel_areas = pixel_areas / total_area

        if (self.pixel_areas.shape[0] != self.npix):
            raise RuntimeError(
                f"self.pixel_areas.shape != self.N, {self.pixel_areas.shape} != {self.npix}")

        self.set_lmn()

    def mesh(self, pts):
        logger.info("Meshing {}".format(pts.shape))
        self.tri = Delaunay(pts)

        # logger.info("Optimizing Mesh {} {}".format(self.tri.points.shape, self.tri.simplices.shape))
        # X, cells = optimesh.cpt.linear_solve_density_preserving(self.tri.points, self.tri.simplices.copy(),
        #  1.0e-10, 100, verbose=True)
        # self.tri = Delaunay(X)
        print(self.tri.simplices[0:10, :])

        self.npix = self.tri.simplices.shape[0]
        logger.info("New Mesh {}".format(self.npix))
        self.pixels = np.zeros(self.npix)

        # Scale points
        self.points = self.radius_rad * \
            np.sum(self.tri.points[self.tri.simplices], axis=1) / 3
        pixel_areas = (
            np.array(
                [area(cell=c, points=self.tri.points)
                 for c in self.tri.simplices]
            )
        )
        total_area = np.sum(pixel_areas)

        logger.info(f"Total area {total_area}")

        self.pixel_areas = pixel_areas / total_area

        if (self.pixel_areas.shape[0] != self.npix):
            raise RuntimeError(
                f"self.pixel_areas.shape != self.N, {self.pixel_areas.shape} != {self.npix}")

        self.set_lmn()

    def gradient(self):
        # Return a gradient between every pair of cells
        gradients = []
        cell_pairs = []

        tri = Delaunay(self.points)

        r_min = self.res_min.radians() / self.fov.radians()
        logger.info(f"Gradient: r_min: {r_min}")

        n_ignored = 0
        for p1, nlist in enumerate(tri.neighbors):
            y1 = self.pixels[p1]
            # print(p1, nlist)
            for p2 in nlist:
                if p2 != -1:
                    dx, dy = self.points[p2] - self.points[p1]
                    r = np.sqrt(dx * dx + dy * dy)
                    if r > r_min:
                        grad = (
                            y1 - self.pixels[p2]
                        ) / r  # TODO Check this division by /r
                        gradients.append([grad, r])
                        cell_pairs.append([p1, p2])
                    else:
                        n_ignored += 1
        logger.info("Gradient Ignored: {} of {} points".format(
            n_ignored, self.npix))

        return np.array(gradients), cell_pairs

    def refine(self):
        grad, pairs = self.gradient()

        self.refine_adding(grad, pairs)

    def refine_adding(self, gradr, pairs):

        logger.info("gradr {}".format(gradr.shape))

        grad = gradr[:, 0]
        rlist = gradr[:, 1]

        p05, p50, p95 = np.percentile(grad, [5, 50, 95])
        logger.info(
            "Grad Percentiles: 5: {} 50: {} 95: {}".format(p05, p50, p95))
        r05, r50, r95 = np.percentile(rlist, [5, 50, 95])
        logger.info("r Percentiles: 5: {} 50: {} 95: {}".format(r05, r50, r95))

        new_pts = self.refine_removing()

        new_count = 0
        for g, r, p in zip(grad, rlist, pairs):
            if g > p95:
                pt = (self.points[p[0]] + self.points[p[1]]) / 2
                new_pts.append(pt)
                new_count += 1

        logger.info(
            "Added {} points to {} -> ".format(new_count,
                                               self.tri.points.shape)
        )

        # self.tri.add_points(new_pts)
        # self.optimize()
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
            # print(p1, nlist)
            g = 0
            edge = False
            for p2 in nlist:
                if p2 != -1:
                    dx, dy = self.points[p2] - self.points[p1]
                    r = np.sqrt(dx * dx + dy * dy)
                    grad = (y1 - self.pixels[p2]) / r  # TODO Fix Gradient /r
                    g += grad * grad
                else:
                    edge = True
            g = np.sqrt(g) / len(nlist)  # average gradient
            gradlist.append(g)
            edgelist.append(edge)

        grad = np.abs(np.array(gradlist))
        p05, p50, p95 = np.percentile(grad, [5, 50, 95])
        logger.info(
            "Grad Percentiles: 5: {} 50: {} 95: {}".format(p05, p50, p95))

        new_indices = []  # Just store the indices
        # Now remove entire cells
        for p, g, e in zip(self.tri.simplices, grad, edgelist):
            if e or (g > p05):
                new_indices += list(
                    p
                )  # ERROR FIXME this adds points multiple times! Only add points sensibly!

        new_indices = np.unique(new_indices)
        logger.info(
            "Removed: {} points to {} -> ".format(
                self.tri.points.shape, new_indices.shape
            )
        )

        return list(self.tri.points[new_indices].copy())

    def plot(self):
        import matplotlib.pyplot as plt

        # plt.plot(pts[:,0], pts[:,1], '.')
        plt.clf()
        plt.plot(self.tri.points[:, 0], self.tri.points[:, 1], "o")
        plt.plot(self.points[:, 0], self.points[:, 1], ".")
        plt.triplot(
            self.tri.points[:, 0], self.tri.points[:,
                                                   1], self.tri.simplices.copy()
        )
        plt.show()

    def callback(self, x, i):
        fname = f"callback_{i:05d}.vtk"
        self.set_visible_pixels(x)
        self.write_mesh(fname)

    def write_mesh(self, fname="output.vtk"):
        # Add a zero third dimension to avoid a VTK warning.
        logger.info(f"Writing VTK file {fname}")
        mesh_pts = np.zeros((self.points.shape[0], 3))
        mesh_pts[:, 0] = self.points[:, 0]
        mesh_pts[:, 1] = self.points[:, 1]

        # Scale mesh points if they're too small
        mesh_pts = mesh_pts * 1000
        # and write it to a file
        meshio.write_points_cells(
            fname,
            mesh_pts,
            [("triangle", self.simplices)],
            cell_data={"flux": [self.pixels]},
        )

    def set_lmn(self):
        x = self.points[:, 0]
        y = self.points[:, 1]
        r = np.sqrt(x * x + y * y)

        # Convert the x,y to theta and phi

        theta = r
        phi = np.arctan2(x, y)

        el_r, az_r = hp2elaz(theta, phi)

        self.el_r = el_r
        self.az_r = az_r

        self.l, self.m, n = elaz2lmn(self.el_r, self.az_r)
        self.n_minus_1 = n - 1


if __name__ == "__main__":

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    fh = logging.FileHandler("disko.log")
    fh.setLevel(logging.INFO)

    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # add formatter to ch
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)

    # add ch to logger
    logger.addHandler(ch)
    logger.addHandler(fh)

    sph = AdaptiveMeshSphere.from_resolution(
        res_arcmin=10,
        res_arcmax=180,
        theta=np.radians(0.0),
        phi=0.0,
        radius=np.radians(20),
    )
    sph.pixels = np.random.random(sph.npix)
    sph.plot()
    sph.refine()
    sph.plot()
    sph.write_mesh()
    sph.to_fits("test.fits", fov=20)
