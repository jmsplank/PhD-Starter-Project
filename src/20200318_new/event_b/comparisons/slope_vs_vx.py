import numpy as np
import matplotlib.pyplot as plt
from phdhelper.helpers import override_mpl
from phdhelper.helpers.os_shortcuts import get_path
import matplotlib

override_mpl.override()

path = get_path(__file__)
path2 = get_path(__file__, "..")

grads = np.load(f"{path2}/mag_spec/grads.npy")
grads_times = np.load(f"{path2}/mag_spec/times.npy")
dt = grads_times[1] - grads_times[0]

# fsm = np.load(f"{path2}/mag_spec/fsm_sampled_100.npy")
vx = np.load(f"{path2}/data/fpi/data_bulkv_i.npy")[:, 0]
time_i = np.load(f"{path2}/data/fpi/time_i.npy")
vx_sampled = np.empty_like(grads_times)


def find_index(arr, val):
    return np.argmin(abs(arr - val))


print(time_i[0])
for i in range(len(grads_times)):
    vx_sampled[i] = vx[
        find_index(time_i, grads_times[i] - dt // 2) : find_index(
            time_i, grads_times[i] + dt // 2
        )
    ].mean()
# vx_sampled = np.abs(vx_sampled)
# print(vx_times, vx_times.shape)

fig, ax = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(6.4, 2.64))

for i in range(3):
    ax[i].hist2d(
        vx_sampled,
        grads[:, i],
        cmin=0.01,
        vmin=1,
        vmax=7,
        bins=16,
        range=(
            (vx_sampled.min() - 40, vx_sampled.max() + 40),
            (grads.min() - 1, grads.max() + 1),
        ),
    )
    ax[i].scatter(
        vx_sampled,
        grads[:, i],
        marker="o",
        edgecolors="k",
        facecolors="none",
        s=12,
        alpha=0.8,
    )
    ax[i].axhline(-5 / 3)
    ax[i].set_xlim(vx_sampled.min() - 40, vx_sampled.max() + 40)
    ax[i].set_ylim(grads.min() - 1, grads.max() + 1)
    ax[i].set_title(["Inertial range", "Ion range", "Electron range"][i])
    ax[i].set(adjustable="box")

ax[0].set_ylabel("Power law fitted slope")
ax[1].set_xlabel("$v_x \quad [kms^{-1}]$")

plt.tight_layout()
plt.subplots_adjust(wspace=0)
plt.savefig(f"{path}/slope_vs_vx.png", dpi=300)
plt.show()