import numpy as np
import matplotlib.pyplot as plt
from phdhelper.helpers import override_mpl
from phdhelper.helpers.os_shortcuts import get_path
from scipy.stats import pearsonr

override_mpl.override()

path = get_path(__file__)
path2 = get_path(__file__, "..")

grads = np.load(f"{path2}/mag_spec/grads.npy")
grads_times = np.load(f"{path2}/mag_spec/times.npy")

fig, ax = plt.subplots(1, 1, sharex=True, sharey=True, figsize=(4, 4))

ax.hist2d(
    grads[:, 0],
    grads[:, 2],
    cmin=0.01,
    vmin=1,
    vmax=7,
    bins=16,
    range=(
        (grads.min() - 1, grads.max() + 1),
        (grads.min() - 1, grads.max() + 1),
    ),
)
ax.scatter(
    grads[:, 0],
    grads[:, 2],
    marker="o",
    edgecolors="k",
    facecolors="none",
    s=30,
    alpha=0.6,
)
ax.axhline(-5 / 3)
ax.axvline(-5 / 3)
ax.set_ylim(grads.min() - 1, grads.max() + 1)
ax.set_xlim(grads.min() - 1, grads.max() + 1)
ax.set_title("Inertial range slope vs. Electron range")

ax.set_ylabel("Electron slope")
ax.set_xlabel("Inertial slope")

print(f"Pearson's correlation coefficient: {pearsonr(grads[:, 0],grads[:, 2])[0]:.3f}")

plt.tight_layout()
plt.savefig(f"{path}/slope_inert_vs_elec.png")
plt.show()