"""Custom interpolations functions.
Split the two major steps, finding grid mappings & interpoalting data,
into 2 functions. Since grid is always the same, there's no need to
repeat this intensive set of steps.
"""
from scipy.spatial import Delaunay
import numpy as np


def interp_weights(xyz, uvw, d=3):
    """Calculate vertices and weights for grid.
    This function, along with interpolate(), splits up the
    functionality of scipy.interpolate.griddata.

    Since the grid is always the same, there's no reason to
    perform the same triangulation every single time, when
    only the values change.

    SRC: https://stackoverflow.com/a/20930910/4465614
    """

    # Do the trinagulation (Delaunay method)
    tri = Delaunay(xyz)

    # Obtain simplices (3d, tetrahedral triangles)
    simplex = tri.find_simplex(uvw)
    # Get the vertices of each simplex
    # vertices[nsimplex, d+1] <- 4 points make up 1 tetrahedron
    vertices = np.take(tri.simplices, simplex, axis=0)
    # Temporary array - intermediate step in obtaining
    # barycentric coordinates
    temp = np.take(tri.transform, simplex, axis=0)
    delta = uvw - temp[:, d]
    # Get barycentric coords of each interp-to point
    bary = np.einsum("njk,nk->nj", temp[:, :d, :], delta)
    # return the vertices & weights of each vertex
    return vertices, np.hstack((bary, 1 - bary.sum(axis=1, keepdims=True)))


def interpolate(values, vtx, wts, fill_value=np.nan):
    # Interpolation step
    ret = np.einsum("nj,nj->n", np.take(values, vtx), wts)
    # If the weight is -ve, then point is outside convex hull
    # replace with fill_value
    ret[np.any(wts < 0, axis=1)] = fill_value
    return ret
