import logging

import numpy as np

import scipy
from sklearn import linear_model


logger = logging.getLogger(__name__)
logger.addHandler(
    logging.NullHandler()
)  # Add other handlers if you're using this as a library
logger.setLevel(logging.INFO)


def plsqr(A, v, alpha, eps=1e-4):
    """
    Solve Ax = v using least squares and subspace projection.
    We'll start using a two-way split of A


    http://www.sam.math.ethz.ch/~mhg/pub/biksm.pdf

    We need to use a block-matrix pseudoinverse
    https://en.wikipedia.org/wiki/Block_matrix_pseudoinverse

    Iterative methods for singular systems.
    https://web.stanford.edu/group/SOL/dissertations/sou-cheng-choi-thesis.pdf

    Even better idea is to use a preconditioned conjugate gradient
    (or other Krylov Subspace technique) where the preconditioner is A^H
    (which is an approximate inverse using the DFT)

    So can use a block matrix pseudoinverse with a preconditioned conjugate gradient
    to solve the problem

    The preconditioner should be A^H which is almost like an inverse (this is equivalent
    to the IFFT based image).

       x = A^H v

    The preconditioned system is

       A^H A x = A^H v

    Where A^H A is symmetric and positive semi-definite.

    Then can use LSQR to do a preconditioned matrix-free method like the Lanctzos algorithm to find
    the singular values, or just solve the lsqr.

    Links:
        Power Iteration (https://en.wikipedia.org/wiki/Power_iteration). Find the largest eigenvalue
        Matrix Free Methods (https://en.wikipedia.org/wiki/Matrix-free_methods)


    """

    print("A = {}".format(A.shape))
    print("v = {}".format(v))
    r = A.shape[1] // 2
    n = A.shape[1]

    A0 = A[:, 0:r]
    A1 = A[:, r:n]

    A0p = np.linalg.pinv(A0)
    A1p = np.linalg.pinv(A1)

    print("A0p = {}".format(A0p.shape))
    print("A1p = {}".format(A1p.shape))

    v0 = v / 2
    v1 = v / 2

    n = 0
    while True:
        reg = (
            linear_model.LinearRegression()
        )  # ElasticNet(alpha=alpha, l1_ratio=0.01, max_iter=10000, positive=False)
        x0 = A0p @ v0
        x1 = A1p @ v1

        v0_1 = A0 @ x0
        v1_1 = A1 @ x1

        r0 = scipy.linalg.norm(v0_1 - v0)
        r1 = scipy.linalg.norm(v1_1 - v1)
        r = scipy.linalg.norm(v0_1 + v1_1 - v)

        print("r0 = {}, r1 = {}, r={}".format(r0, r1, r))

        if r < eps:
            break

        n = n + 1
        if n > 50:
            raise RuntimeError("Too many iterations")

        v0 = v0_1
        v1 = v - v0

    return np.block([x0, x1])


if __name__ == "__main__":
    A = np.random.random((10, 8))
    x = np.random.random(8)

    v = A @ x
    print(x)

    reg = linear_model.LinearRegression()
    reg.fit(A, v)
    x = reg.coef_
    print(x)
    r = scipy.linalg.norm(A @ x - v)
    print("Residual {}".format(r))

    x = plsqr(A, v, alpha=0.0)
    print(x)
