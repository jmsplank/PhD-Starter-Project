import numpy as np
import matplotlib.pyplot as plt
import os
from phdhelper.helpers import override_mpl
from phdhelper.helpers.COLOURS import red, green
from datetime import datetime as dt
import matplotlib

override_mpl.override()
matplotlib.rc("font", family="eb garamond")
matplotlib.rc("text", usetex=True)
matplotlib.rc(
    "text.latex",
    preamble=r"""\usepackage[T1]{fontenc}
\usepackage{ebgaramond}
\usepackage{bm}""",
)
matplotlib.rcParams.update({"font.size": 10})

path = os.path.dirname(os.path.realpath(__file__))
dirpath = "/".join(path.split("/")[:-1])

big_B = np.load(f"{dirpath}/data/fgm/data.npy")[:, 3]
big_B_time = np.load(f"{dirpath}/data/fgm/time.npy")

shock = 1584500550.1
shock_dt = dt.utcfromtimestamp(shock)
print(shock_dt)

grads = np.load(f"{path}/grads.npy")
times = np.load(f"{path}/times.npy")

fig, ax = plt.subplots(3, 1, sharey=True, figsize=(6, 6))

for i in range(3):
    sheath = grads[times < shock, i]
    sw = grads[times >= shock, i]
    ax[i].hist(sheath, label="STR", alpha=0.8)
    ax[i].hist(sw, label="SW", alpha=0.8)
    ax[i].set_title(["Intertial", "Ion", "Electron"][i])
    ax[i].set_ylabel("Counts")
    ax[-1].set_xlabel("Slope")
    ax[i].set_xlim((-4.25, 0.25))
    ax[i].vlines(
        sheath.mean(),
        0,
        14,
        colors="k",
        linestyles="solid",
        label=rf"$\left<STR\right> = {sheath.mean():0.2f} \pm {sheath.std():0.2f}$",
    )
    ax[i].vlines(
        sw.mean(),
        0,
        14,
        colors="k",
        linestyles="dashed",
        label=rf"$\left<SW\right> = {sw.mean():0.2f} \pm {sw.std():0.2f}$",
    )
    ax[i].legend()

plt.tight_layout()
plt.show()