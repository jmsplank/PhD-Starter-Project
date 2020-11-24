"""Comparison of results from scipy.griddate & custom interpol.
Ouputs two plots with triangulations overlaid.
Ideally would be the same if interpolation methods work correctly.
"""
import numpy as np
import itertools
from scipy.interpolate import griddata, LinearNDInterpolator
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt

import testing_fastInterp as tfa
from interpol import *

ni, d = int(30), 2
points = np.random.rand(ni, d)
values = np.random.rand(ni)

nf = 1000
x, y = np.mgrid[0:1:1000j, 0:1:1000j]
xy = np.array([x.flatten(), y.flatten()]).T

data = griddata(points, values, xy).reshape((nf, nf))

print(xy.shape)
vtx, wts = interp_weights(points, xy, d=2)
data2 = tfa.interpolate(values, vtx, wts).reshape((nf, nf))
# data2[~np.isnan(data2)] = 0.0
# data2[np.isnan(data2)] = 1.0

tri = Delaunay(points)
# itrp = LinearNDInterpolator(tri, values)
# data2 = itrp(xy).reshape((nf, nf))

print(data.shape)
print(data2.shape)

fig, ax = plt.subplots(2, 1)

ax[0].pcolormesh(x, y, data, shading="auto")
ax[0].triplot(points[:, 0], points[:, 1], tri.simplices)
ax[1].pcolormesh(x, y, data2, shading="auto")
ax[1].triplot(points[:, 0] + 1.0 / nf, points[:, 1] + 1.0 / nf, tri.simplices)
plt.show()
