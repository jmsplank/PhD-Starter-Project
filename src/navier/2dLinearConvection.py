from mpl_toolkits.mplot3d import axes3d

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

nx = 81
ny = 81
nt = 100
c = 1
dx = 2 / (nx - 1)
dy = 2 / (ny - 1)
sigma = 0.2
dt = sigma * dx

x = np.linspace(0, 2, nx)
y = np.linspace(0, 2, ny)

u = np.ones((ny, nx))
u[int(0.5 / dy) : int(1 / dy + 1), int(0.5 / dx) : int(1 / dx + 1)] = 2

for n in range(nt + 1):
    un = u.copy()
    u[1:, 1:] = (
        un[1:, 1:]
        - (c * dt / dx * (un[1:, 1:] - un[1:, :-1]))
        - (c * dt / dy * (un[1:, 1:] - un[:-1, 1:]))
    )

    u[0, :] = 1
    u[-1, :] = 1
    u[:, 0] = 1
    u[:, -1] = 1

fig = plt.figure(figsize=(11, 7), dpi=100)
ax = fig.gca(projection="3d")
X, Y = np.meshgrid(x, y)
surf = ax.plot_surface(X, Y, u[:], cmap=cm.viridis)
plt.show()