"""
SRC:
https://nbviewer.jupyter.org/github/barbagroup/CFDPython/blob/master/lessons/01_Step_1.ipynb
"""
import numpy as np
import matplotlib.pyplot as plt
import time, sys

nx = 41
dx = 2.0 / (nx - 1)
nt = 20
dt = 0.025

u = np.ones(nx)
u[int(0.5 / dx) : int(1 / dx + 1)] = 2

for n in range(nt):
    un = u.copy()
    for i in range(1, nx):
        u[i] = un[i] - un[i] * dt / dx * (un[i] - un[i - 1])

plt.plot(np.linspace(0, 2, nx), u)
plt.show()
