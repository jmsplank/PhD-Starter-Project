from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np

###variable declarations
nx = 101
ny = 101
nt = 80
c = 1
dx = 2 / (nx - 1)
dy = 2 / (ny - 1)
sigma = 0.2
dt = sigma * dx

x = np.linspace(0, 2, nx)
y = np.linspace(0, 2, ny)

u = np.ones((ny, nx))  ##create a 1xn vector of 1's
v = np.ones((ny, nx))
un = np.ones((ny, nx))
vn = np.ones((ny, nx))

###Assign initial conditions
##set hat function I.C. : u(.5<=x<=1 && .5<=y<=1 ) is 2
u[int(0.5 / dy) : int(1 / dy + 1), int(0.5 / dx) : int(1 / dx + 1)] = 2
##set hat function I.C. : v(.5<=x<=1 && .5<=y<=1 ) is 2
v[int(0.5 / dy) : int(1 / dy + 1), int(0.5 / dx) : int(1 / dx + 1)] = 2

for n in range(nt + 1):
    un = u.copy()
    vn = v.copy()

    u[1:, 1:] = (
        un[1:, 1:]
        - (un[1:, 1:] * c * dt / dx * (un[1:, 1:] - un[1:, :-1]))
        - vn[1:, 1:] * c * dt / dy * (un[1:, 1:] - un[:-1, 1:])
    )
    v[1:, 1:] = (
        vn[1:, 1:]
        - (un[1:, 1:] * c * dt / dx * (vn[1:, 1:] - vn[1:, :-1]))
        - vn[1:, 1:] * c * dt / dy * (vn[1:, 1:] - vn[:-1, 1:])
    )

    u[0, :] = 1
    u[-1, :] = 1
    u[:, 0] = 1
    u[:, -1] = 1

    v[0, :] = 1
    v[-1, :] = 1
    v[:, 0] = 1
    v[:, -1] = 1


fig = plt.figure()
ax = fig.gca(projection="3d")
X, Y = np.meshgrid(x, y)

ax.plot_surface(X, Y, u, cmap=cm.viridis, rstride=2, cstride=2)
ax.set_xlabel("$x$")
ax.set_ylabel("$y$")
plt.show()
