from datetime import datetime as dt
from datetime import timedelta

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np

from phdhelper.suMMSary import Event
from phdhelper.helpers import format_timestamps
import phdhelper.helpers.override_mpl as ovr

ovr.override()

trange = ["2020-03-18/02:05:00", "2020-03-18/02:44:00"]
probe = "1"

summary = Event(trange=trange, probe=probe, required_instruments="FGM|FPI")
print(summary)


fig, ax = plt.subplots(6, 2, gridspec_kw={"width_ratios": [98, 3]})


ax[0, 0].plot(summary.B.value[:, 3], lw=1)
ax[0, 0].set_ylabel("$|B|\quad (nT)$")

ax[1, 0].plot(summary.B.value[:, :3], lw=1)
ax[1, 0].set_ylabel("$B\quad (nT)$")

ax[2, 0].plot(summary.v.ion.value, lw=1)
ax[2, 0].set_ylabel("$V_i\quad (kms^{-1})$")


ax[3, 0].plot(summary.T.ion.value)
ax[3, 0].set_ylabel("$T_i \quad (MK)$")

# ax[4, 0].plot(summary.T.electron.time.date_time, summary.T.electron.value)
# ax[4, 0].set_ylabel("$T_e \quad (MK)$")

data_E = summary.E.ion.value
data_Et = summary.E.ion.time.timestamp
e_bins = np.logspace(np.log10(2), np.log10(17800), 32)
t_bins = np.linspace(data_Et.min(), data_Et.max(), len(data_Et))
im = ax[5, 0].pcolormesh(
    t_bins,
    e_bins,
    data_E.T,
    shading="nearest",
    norm=colors.LogNorm(vmin=data_E.min() + 100, vmax=data_E.max()),
)
ax[5, 0].set_yscale("log")
ax[5, 0].set_ylabel("log $E_i\quad (eV)$")
fig.colorbar(im, cax=ax[5, 1])

# Convert timestamps to formatted labels
# times_locs, times_names = format_timestamps(tick_start="2020-03-18/02:26:00")
# for i in range(6):
#     # Explicitly set tick locations
#     ax[i, 0].set_xticks(times_locs)
#     # Set tick labels
#     ax[i, 0].set_xticklabels(times_names)

# Turn off unused axes
for i in range(5):
    ax[i, 1].axis("off")

# Share X axis
ax[0, 0].get_shared_x_axes().join(*[ax[i, 0] for i in range(6)])

# Turn off x labels for all plots above bottom
for i in range(5):
    ax[i, 0].tick_params(
        axis="x", which="both", bottom="on", top="on", labelbottom=False
    )

plt.tight_layout(
    pad=0.1,
)
plt.show()
