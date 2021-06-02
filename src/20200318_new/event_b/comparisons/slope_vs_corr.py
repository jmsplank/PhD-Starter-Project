import numpy as np
import matplotlib.pyplot as plt
from phdhelper.helpers import override_mpl
from phdhelper.helpers.os_shortcuts import get_path

override_mpl.override()

path = get_path(__file__)
path2 = get_path(__file__, "..")

grads = np.load(f"{path2}/mag_spec/grads.npy")
grads_times = np.load(f"{path2}/mag_spec/times.npy")

corr = np.load(f"{path2}/correlation/corr_lens.npy")
corr_times = np.load(f"{path2}/correlation/corr_times.npy")

fig, ax = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(10, 4))

for i in range(3):
    ax[i].hist2d(
        corr,
        grads[:, i],
        cmin=0.01,
        vmin=1,
        vmax=6,
        bins=16,
        range=(
            (corr.min() - 4, corr.max() + 4),
            (grads.min() - 1, grads.max() + 1),
        ),
    )
    ax[i].scatter(
        corr,
        grads[:, i],
        marker="o",
        edgecolors="k",
        facecolors="none",
        s=30,
        alpha=0.6,
    )
    ax[i].axhline(-5 / 3)
    ax[i].set_xlim(corr.min() - 4, corr.max() + 4)
    ax[i].set_ylim(grads.min() - 1, grads.max() + 1)
    ax[i].set_title(["Inertial range", "Ion range", "Electron range"][i])

ax[0].set_ylabel("Power law fitted slope")
ax[1].set_xlabel("Correlation length $[d_i]$")

plt.tight_layout()
plt.savefig(f"{path}/slope_vs_corr.png")
plt.show()