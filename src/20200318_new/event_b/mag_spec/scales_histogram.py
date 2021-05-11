import numpy as np
import matplotlib.pyplot as plt
import os
from phdhelper.helpers import override_mpl
from phdhelper.helpers.COLOURS import red, green
from datetime import datetime as dt

override_mpl.override()

path = os.path.dirname(os.path.realpath(__file__))
dirpath = "/".join(path.split("/")[:-1])

big_B = np.load(f"{dirpath}/data/fgm/data.npy")[:, 3]
big_B_time = np.load(f"{dirpath}/data/fgm/time.npy")

shock = 1584500550.1
shock_dt = dt.utcfromtimestamp(shock)
print(shock_dt)

grads = np.load(f"{path}/grads.npy")
times = np.load(f"{path}/times.npy")

fig, ax = plt.subplots(1, 3, sharey=True)

for i in range(3):
    sheath = grads[times < shock, i]
    sw = grads[times >= shock, i]
    ax[i].hist(sheath, label="Sheath", alpha=0.8)
    ax[i].hist(sw, label="SW", alpha=0.8)
    ax[i].set_title(["Intertial", "Ion", "Electron"][i])
    ax[i].vlines(
        sheath.mean(),
        0,
        14,
        colors="k",
        linestyles="solid",
        label=f"<sheath> = {sheath.mean():0.2f} +/- {sheath.std():0.2f}",
    )
    ax[i].vlines(
        sw.mean(),
        0,
        14,
        colors="k",
        linestyles="dashed",
        label=f"<SW> = {sw.mean():0.2f} +/- {sw.std():0.2f}",
    )
    ax[i].legend()

plt.show()