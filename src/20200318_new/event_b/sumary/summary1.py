import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from phdhelper.helpers import override_mpl, os_shortcuts
from datetime import datetime as dt

override_mpl.override()
matplotlib.rc("font", family="eb garamond")
matplotlib.rc("text", usetex=True)
matplotlib.rc(
    "text.latex",
    preamble=r"""\usepackage[T1]{fontenc}
\usepackage{ebgaramond}
\usepackage{bm}""",
)
matplotlib.rcParams.update({"font.size": 12})

path = os_shortcuts.new_path(os_shortcuts.get_path(__file__, ".."))
savepath = os_shortcuts.new_path(os_shortcuts.get_path(__file__))

B = np.load(f"{path('data/fgm')}/data.npy")
B_time = np.load(f"{path('data/fgm')}/time.npy")

i_time = np.load(f"{path('data/fpi')}/time_i.npy")

e_numdens = np.load(f"{path('data/fpi')}/data_numberdensity_e.npy")
e_numdens_time = np.load(f"{path('data/fpi')}/time_numberdensity_e.npy")

i_numdens = np.load(f"{path('data/fpi')}/data_numberdensity_i.npy")

e_tempperp = np.load(f"{path('data/fpi')}/data_tempperp_e.npy")
e_tempperp_time = np.load(f"{path('data/fpi')}/time_tempperp_e.npy")

i_tempperp = np.load(f"{path('data/fpi')}/data_tempperp_i.npy")

e_temppara = np.load(f"{path('data/fpi')}/data_temppara_e.npy")
e_temppara_time = np.load(f"{path('data/fpi')}/time_temppara_e.npy")

i_temppara = np.load(f"{path('data/fpi')}/data_temppara_i.npy")

e_bulkv = np.load(path("data/fpi/data_bulkv_e.npy"))
e_bulkv_time = np.load(path("data/fpi/time_bulkv_e.npy"))


fig, ax = plt.subplots(6, 1, sharex=True, figsize=(6.69, 8))


ax[0].plot(B_time, B[:, 3], color="k")
for i in range(3):
    ax[1].plot(B_time, B[:, i], label=["x", "y", "z"][i])
    ax[2].plot(e_bulkv_time, e_bulkv[:, i], label=["x", "y", "z"][i])
# ax[2].plot(e_bulkv_time, np.linalg.norm(e_bulkv, axis=1), label=r"$\left<v_i\right>$")
ax[3].plot(e_numdens_time, e_numdens, label="electron")
ax[3].plot(i_time, i_numdens, label="ion")
ax[4].plot(e_tempperp_time, e_tempperp, label="$T_{\perp}$")
ax[4].plot(e_temppara_time, e_temppara, label="$T_{||}$")
ax[5].plot(i_time, i_tempperp, label="$T_{\perp}$")
ax[5].plot(i_time, i_temppara, label="$T_{||}$")

ax[0].set_ylabel("$|B|$ $[nT]$")
ax[1].set_ylabel(r"$\bm{B}$ $[nT]$")
ax[2].set_ylabel(r"$v_e$ $[kms^{-1}]$")
ax[3].set_ylabel("Number density $[cm^{-3}]$")
ax[4].set_ylabel("$T_e$ $[MK]$")
ax[5].set_ylabel("$T_i$ $[MK]$")

for i in [1, 2, 3, 4, 5]:
    ax[i].legend(loc="upper right")
ax[-1].set_xlabel("Time UTC 2020/03/18 (hh:mm:ss) ")
ax[-1].set_xticklabels(
    map(lambda x: dt.strftime(dt.utcfromtimestamp(x), "%H:%M:%S"), ax[3].get_xticks())
)

plt.tight_layout()
# plt.show()
plt.savefig(savepath("summary1.png"), dpi=300)
