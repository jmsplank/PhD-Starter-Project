from datetime import datetime as dt
from datetime import timedelta
import pyspedas
from pytplot import data_quants

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np

from phdhelper.suMMSary import Event
from phdhelper.helpers import format_timestamps
import phdhelper.helpers.override_mpl as ovr

ovr.override()

# trange = ["2020-03-18/02:25:30", "2020-03-18/02:44:00"]
trange = ["2020-03-18/02:05:00", "2020-03-18/02:45:00"]
probe = "1"

# summary = Event(trange=trange, probe=probe, required_instruments="FGM|FPI")
pyspedas.mms.fgm(trange=trange, probe=probe, data_rate="brst")
pyspedas.mms.fpi(trange=trange, probe=probe, data_rate="brst")

B = data_quants["mms1_fgm_b_gse_brst_l2"].values
B_t = data_quants["mms1_fgm_b_gse_brst_l2"].coords["time"].values

v = data_quants["mms1_dis_bulkv_gse_brst"].values
v_t = data_quants["mms1_dis_bulkv_gse_brst"].coords["time"].values

T_perp = data_quants["mms1_dis_tempperp_brst"].values
T_t = data_quants["mms1_dis_tempperp_brst"].coords["time"].values
T_para = data_quants["mms1_dis_temppara_brst"].values

T_e_perp = data_quants["mms1_des_tempperp_brst"].values
T_t_e = data_quants["mms1_des_tempperp_brst"].coords["time"].values
T_e_para = data_quants["mms1_des_temppara_brst"].values

T_i = np.column_stack((T_perp, T_para))
T_e = np.column_stack((T_e_perp, T_e_para))

E = data_quants["mms1_dis_energyspectr_omni_brst"].values
E_t = data_quants["mms1_dis_energyspectr_omni_brst"].coords["time"].values

fig, ax = plt.subplots(6, 2, gridspec_kw={"width_ratios": [98, 3]})


ax[0, 0].plot(B_t, B[:, 3], lw=1)
ax[0, 0].set_ylabel("$|B|\quad (nT)$")

ax[1, 0].plot(B_t, B[:, :3], lw=1)
ax[1, 0].set_ylabel("$B\quad (nT)$")

ax[2, 0].plot(v_t, v, lw=1)
ax[2, 0].set_ylabel("$V_i\quad (kms^{-1})$")


ax[3, 0].plot(T_t, T_i)
ax[3, 0].set_ylabel("$T_i \quad (MK)$")

ax[4, 0].plot(T_t_e, T_e)
ax[4, 0].set_ylabel("$T_e \quad (MK)$")

e_bins = np.logspace(np.log10(2), np.log10(17800), 32)
t_bins = np.linspace(E_t.min(), E_t.max(), len(E_t))
im = ax[5, 0].pcolormesh(
    t_bins,
    e_bins,
    E.T,
    shading="nearest",
    norm=colors.LogNorm(vmin=E.min() + 100, vmax=E.max()),
)
ax[5, 0].set_yscale("log")
ax[5, 0].set_ylabel("log $E_i\quad (eV)$")
fig.colorbar(im, cax=ax[5, 1])

# Convert timestamps to formatted labels
times_locs, times_names = format_timestamps(
    tick_start="2020-03-18/02:05:00", tick_step=("minutes", 5), num=9
)
for i in range(6):
    # Explicitly set tick locations
    ax[i, 0].set_xticks(times_locs)
    # Set tick labels
    ax[i, 0].set_xticklabels(times_names)

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
