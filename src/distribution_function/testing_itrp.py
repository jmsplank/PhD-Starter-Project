"""Demonstration of interpolation accuracy in 
replicating data observation.
"""
import numpy as np
from scipy import interpolate
from scipy.spatial import Delaunay
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt


def func(x, y):
    return x * (1 - x) * np.cos(4 * np.pi * x) * np.sin(4 * np.pi * y) ** 2


points = np.random.random((400, 2))
tri = Delaunay(points)

values = func(points[:, 0], points[:, 1])

itrp_f = interpolate.LinearNDInterpolator(points, values, fill_value=0.0)

num = 512 ** 2
sqrt_num = int(np.sqrt(num))
xy = np.arange(0, 1, 1.0 / sqrt_num)
XY = np.meshgrid(xy, xy)
new_coords = np.array(list(zip(XY[0].flatten(), XY[1].flatten())))

new_values = np.array(itrp_f(new_coords)).reshape((sqrt_num, sqrt_num))
# new_values = gaussian_filter(new_values, sigma=5)

grid = interpolate.griddata(points, values, (XY[0], XY[1]), method="cubic")

fig, ax = plt.subplots(1, 2)

# ax[0].pcolormesh(
#     *[np.linspace(0 - 1. / sqrt_num, 1 - 1. / sqrt_num, sqrt_num)] * 2,
#     new_values,
#     shading='nearest')

ax[0].imshow(func(XY[0], XY[1]), extent=(0, 1, 0, 1), origin="lower")
ax[0].triplot(points[:, 0], points[:, 1], tri.simplices, c="k", alpha=0.2)
ax[0].plot(points[:, 0], points[:, 1], "o", c="k", alpha=0.2, ms=3)

ax[0].set_xlim(0, 1)
ax[0].set_ylim(0, 1)

ax[1].imshow(grid, extent=(0, 1, 0, 1), origin="lower")

plt.show()
