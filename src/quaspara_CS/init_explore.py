from datetime import datetime as dt
from datetime import timedelta
from pprint import pprint

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import pyspedas
from mpl_toolkits.axes_grid1 import make_axes_locatable
from numpy.lib.arraypad import pad
from phdhelper.suMMSary import EventSummary
from pytplot import data_quants

trange = ["2020-03-18/02:25:30", "2020-03-18/02:44:00"]
probe = "1"

summary = EventSummary(trange=trange, probe=probe)
print(summary)

times = [dt.strptime("2020-03-18/02:26:00", "%Y-%m-%d/%H:%M:%S")]
for i in range(10):
    times.append(times[i] + timedelta(minutes=2))
times2 = [dt.strftime(i, "%H:%M") for i in times]
timeLocs = [dt.timestamp(t) for t in times]

fig, ax = plt.subplots(4, 2, gridspec_kw={"width_ratios": [95, 5]})

for i in range(4):
    ax[i, 0].set_xticks(timeLocs)
    ax[i, 0].set_xticklabels(times2)

for i in range(3):
    ax[i, 1].axis("off")

ax[0, 0].plot(summary.time_B, summary.B[:, 3], lw=1)
ax[0, 0].set_ylabel("$|B|\quad (nT)$")
# ax[0, 0].tick_params(axis="x", which="both", bottom="on", top="on", labelbottom="off")

ax[1, 0].plot(summary.time_B, summary.B[:, :3], lw=1)
ax[1, 0].set_ylabel("$B\quad (nT)$")
# ax[1, 0].tick_params(axis="x", which="both", bottom="on", top="on", labelbottom="off")

ax[2, 0].plot(summary.time_V, summary.v_i, lw=1)
ax[2, 0].set_ylabel("$V_i\quad (kms^{-1})$")
# ax[2, 0].tick_params(axis="x", which="both", bottom="on", top="on", labelbottom="off")

data_E = summary.E_i

data_Et = summary.time_E
e_bins = np.logspace(np.log10(2), np.log10(17800), 32)
t_bins = np.linspace(data_Et.min(), data_Et.max(), len(data_Et))
im = ax[3, 0].pcolormesh(
    t_bins,
    e_bins,
    data_E.T,
    cmap="rainbow",
    shading="nearest",
    norm=colors.LogNorm(vmin=data_E.min() + 1e-29, vmax=data_E.max()),
)
ax[3, 0].set_yscale("log")
ax[3, 0].set_ylabel("log $E_i\quad (eV)$")

fig.colorbar(im, cax=ax[3, 1])

plt.tight_layout()
plt.show()
