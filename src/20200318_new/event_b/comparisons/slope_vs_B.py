import numpy as np
import matplotlib.pyplot as plt
from phdhelper.helpers import override_mpl
from phdhelper.helpers.os_shortcuts import get_path
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
matplotlib.rcParams.update({"font.size": 11})

path = get_path(__file__)
path2 = get_path(__file__, "..")

grads = np.load(f"{path2}/mag_spec/grads.npy")
grads_times = np.load(f"{path2}/mag_spec/times.npy")

fsm = np.load(f"{path2}/mag_spec/fsm_sampled_100.npy")

fig, ax = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(6.6, 2.64))

for i in range(3):
    ax[i].hist2d(
        fsm,
        grads[:, i],
        cmin=0.01,
        vmin=1,
        vmax=10,
        bins=16,
        range=(
            (fsm.min() - 4, fsm.max() + 4),
            (grads.min() - 1, grads.max() + 1),
        ),
    )
    ax[i].scatter(
        fsm,
        grads[:, i],
        marker="o",
        edgecolors="k",
        facecolors="none",
        s=12,
        alpha=0.8,
    )
    ax[i].axhline(-5 / 3)
    ax[i].set_xlim(fsm.min() - 4, fsm.max() + 4)
    ax[i].set_ylim(grads.min() - 1, grads.max() + 1)
    ax[i].set_title(["Inertial range", "Ion range", "Electron range"][i])

ax[0].set_ylabel("Power law fitted slope")
ax[1].set_xlabel("$|B|$ $[nT]$")

plt.tight_layout()
plt.savefig(f"{path}/slope_vs_b.png", dpi=300)
# plt.show()