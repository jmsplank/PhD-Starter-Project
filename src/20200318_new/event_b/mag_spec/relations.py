import numpy as np
import matplotlib.pyplot as plt
import os
from phdhelper.helpers import override_mpl
import matplotlib.colors as mpl_color

override_mpl.override()

path = os.path.dirname(os.path.realpath(__file__))
dirpath = "/".join(path.split("/")[:-1])

grads = np.load(f"{path}/grads.npy")
grads_times = np.load(f"{path}/times.npy")

mag = np.load(f"{dirpath}/data/fgm/data.npy")[:, 3]
mag_time = np.load(f"{dirpath}/data/fgm/time.npy")
mag_rescaled = np.empty(len(grads_times), dtype=float)

grad_time_diff = grads_times[1] - grads_times[0]
for i, t in enumerate(range(len(grads_times))):
    mag_rescaled[i] = mag[
        (mag_time >= grads_times[i]) & (mag_time < grads_times[i] + grad_time_diff)
    ].mean()


fig, ax = plt.subplots(
    1,
    4,
    gridspec_kw={"width_ratios": [32, 32, 32, 4]},
)

for i in range(3):
    im = ax[i].hist2d(
        grads[:, i], mag_rescaled, norm=mpl_color.LogNorm(vmin=1, vmax=15)
    )

fig.colorbar(im[3], cax=ax[3])
ax[0].set_xlim((grads.min(), grads.max()))

for i in range(1, 3):
    print(i)
    ax[i].sharex(ax[i - 1])
    ax[i].sharey(ax[i - 1])

ax[0].set_ylabel("|B| [nT]")
ax[1].set_xlabel("Slope")

plt.tight_layout()
plt.show()
