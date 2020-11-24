"""Low-res demo of how the fast interpol method works
"""
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from scipy.spatial import Delaunay
import inspect


def prnt(*args):
    def retrieve_name(var):
        callers_local_vars = inspect.currentframe().f_back.f_back.f_back
        if callers_local_vars is None:
            return None
        callers_local_vars = callers_local_vars.f_locals.items()
        for var_name, var_val in callers_local_vars:
            if var_val is var:
                return var_name
        return None

    names = [retrieve_name(arg) for arg in args]
    if None in names:
        print("Some args have no variable name. Are they objects?")
    for i, arg in enumerate(args):
        if names[i] is None:
            print(f"ARG{i:03d}:")
        else:
            print(f"{names[i]}:")
        print(arg)
        print("")


def interp_weights(xyz, uvw, d=3):
    tri = Delaunay(xyz)
    simplex = tri.find_simplex(uvw)
    vertices = np.take(tri.simplices, simplex, axis=0)
    print(vertices.shape)
    print(f"vertices.shape: {vertices.shape}")
    temp = np.take(tri.transform, simplex, axis=0)
    print(f"temp.shape: {temp.shape}")
    delta = uvw - temp[:, d]
    print(f"delta.shape: {delta.shape}")
    bary = np.einsum("njk,nk->nj", temp[:, :d, :], delta)
    print(f"bary.shape: {bary.shape}")
    return vertices, np.hstack((bary, 1 - bary.sum(axis=1, keepdims=True)))


def interpolate(values, vtx, wts, fill_value=np.nan):
    ret = np.einsum("nj,nj->n", np.take(values, vtx), wts)
    ret[np.any(wts < 0, axis=1)] = fill_value
    return ret


if __name__ == "__main__":
    ndim = 2
    points = np.array([[1, 0], [1, 2], [0, 0]])
    plt.scatter(points[:, 0], points[:, 1], c="r", marker="x")

    tri = Delaunay(points)
    simplices = tri.simplices
    plt.triplot(points[:, 0], points[:, 1], tri.simplices)

    prnt(points, simplices, tri)

    xx, yy = np.mgrid[0:1:4j, 0:2:8j]
    new_points = np.array([xx.flatten(), yy.flatten()]).T
    prnt(new_points, new_points.shape)

    plt.scatter(new_points[:, 0], new_points[:, 1], c="b", marker="x")
    for i, point in enumerate(new_points):
        plt.text(point[0] + 0.01, point[1] + 0.01, str(i))

    simplex = tri.find_simplex(new_points)
    prnt(simplex)

    transform = tri.transform
    transform_shape = transform.shape
    prnt(transform, transform_shape)

    # Affine transformation T c = x - r
    # transform [nsimplex, ndim+1, ndim]
    # transform[i, :ndim, :ndim] contains T^-1
    T_inv = transform[0, :ndim, :ndim]
    # transform[i, ndim, :] contains r
    r = transform[0, ndim, :]

    prnt(T_inv, r)

    c = T_inv.dot(np.transpose(points - r))
    b = np.c_[c.T, 1 - c.sum(axis=0)]

    b_verts, b_new_points = interp_weights(points, new_points, d=2)

    values = np.array([1, 2, 0])
    point_coord = (np.take(values, b_verts) * b_new_points).sum(axis=1)
    test_interpol = interpolate(values, b_verts, b_new_points)
    prnt(point_coord, test_interpol)

    plt.pcolormesh(
        np.linspace(-0.125, 1.125, 5),
        np.linspace(-0.125, 2.125, 9),
        test_interpol.reshape((4, 8)).T,
    )

    plt.show()
