import logging

import numpy as np
import dask.array as da

logger = logging.getLogger(__name__)
logger.addHandler(
    logging.NullHandler()
)  # Add other handlers if you're using this as a library
logger.setLevel(logging.INFO)


def log_array(title, x):
    if isinstance(x, np.ndarray):
        logger.info("   np: {}:{} {:5.2f} GB".format(title, x.shape, x.nbytes / 1e9))
    elif isinstance(x, da.core.Array):
        logger.info("   da: {}:{} chunks={}".format(title, x.shape, x.chunks))
    else:
        logger.info("   ?: {}:{} {:5.2f} GB".format(title, x.shape, x.nbytes / 1e9))


def da_identity(d, chunks="auto"):
    return da.diag(np.ones(d)).rechunk(chunks)


def da_diagsvd(s, M, N):
    """
    Construct the sigma matrix in SVD from singular values and size M, N.
    Parameters
    ----------
    s : (M,) or (N,) array_like
        Singular values
    M : int
        Size of the matrix whose singular values are `s`.
    N : int
        Size of the matrix whose singular values are `s`.
    Returns
    -------
    S : (M, N) ndarray
        The S-matrix in the singular value decomposition
    """
    part = da.diag(s)

    MorN = len(s)
    if MorN == M:
        return da.block([part, da.zeros((M, N - M), dtype=s.dtype)])
    elif MorN == N:
        return da.block([[part], [da.zeros((M - N, N), dtype=s.dtype)]])
    else:
        raise ValueError("Length of s must be M or N.")


def da_block_diag(*arrs):
    """
    Create a block diagonal matrix from provided arrays.
    Given the inputs `A`, `B` and `C`, the output will have these
    arrays arranged on the diagonal::
        [[A, 0, 0],
         [0, B, 0],
         [0, 0, C]]
    Parameters
    ----------
    A, B, C, ... : array_like, up to 2-D
        Input arrays.  A 1-D array or array_like sequence of length `n` is
        treated as a 2-D array with shape ``(1,n)``.
    Returns
    -------
    D : ndarray
        Array with `A`, `B`, `C`, ... on the diagonal. `D` has the
        same dtype as `A`.
    """
    shapes = np.array([a.shape for a in arrs])
    out_dtype = np.find_common_type([arr.dtype for arr in arrs], [])
    out = da.zeros(np.sum(shapes, axis=0), dtype=out_dtype)
    r, c = 0, 0
    for i, (rr, cc) in enumerate(shapes):
        out[r: r + rr, c: c + cc] = arrs[i]
        r += rr
        c += cc
    return out
