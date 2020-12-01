import numpy
from matplotlib import pyplot, cm
from mpl_toolkits.mplot3d import Axes3D  ##library for 3d projection plots


###variable declarations
nx = 31
ny = 31
nt = 17
nu = 0.05
dx = 2 / (nx - 1)
dy = 2 / (ny - 1)
sigma = 0.25
dt = sigma * dx * dy / nu

x = numpy.linspace(0, 2, nx)
y = numpy.linspace(0, 2, ny)

u = numpy.ones((ny, nx))  # create a 1xn vector of 1's
un = numpy.ones((ny, nx))

###Assign initial conditions
# set hat function I.C. : u(.5<=x<=1 && .5<=y<=1 ) is 2
u[int(0.5 / dy) : int(1 / dy + 1), int(0.5 / dx) : int(1 / dx + 1)] = 2

X, Y = numpy.meshgrid(x, y)
###Run through nt timesteps
def diffuse(nt):
    u[int(0.5 / dy) : int(1 / dy + 1), int(0.5 / dx) : int(1 / dx + 1)] = 2

    for n in range(nt + 1):
        un = u.copy()
        u[1:-1, 1:-1] = (
            un[1:-1, 1:-1]
            + nu * dt / dx ** 2 * (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, 0:-2])
            + nu * dt / dy ** 2 * (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[0:-2, 1:-1])
        )
        u[0, :] = 1
        u[-1, :] = 1
        u[:, 0] = 1
        u[:, -1] = 1

    fig = pyplot.figure()
    ax = fig.gca(projection="3d")
    surf = ax.plot_surface(
        X, Y, u[:], rstride=1, cstride=1, cmap=cm.viridis, linewidth=0, antialiased=True
    )
    ax.set_zlim(1, 2.5)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    pyplot.show()


diffuse(10)
diffuse(14)
diffuse(50)