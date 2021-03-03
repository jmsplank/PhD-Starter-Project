from pyqtgraph.metaarray.MetaArray import axis
import pyspedas
from pytplot import data_quants
from scipy.stats import kurtosis
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime as dt

trange = ["2016-12-09/09:01:36", "2016-12-09/09:07:00"]  # Interval 1
probe = "1"
pyspedas.mms.fsm(trange=trange, probe=probe, time_clip=True, level="l3")

fsm_B = data_quants["mms1_fsm_b_gse_brst_l3"].values
time_dist = data_quants["mms1_fsm_b_gse_brst_l3"].coords["time"].values

windows = 35
width = len(time_dist) // windows
# width = int(1e5)

kurtosisss = np.empty((windows, 3), dtype=float)
x = []
for i in range(windows):
    data = fsm_B[i * width : width * (i + 1), :]
    kurtosisss[i, :] = kurtosis(data, axis=0, fisher=False)
    x.append(dt.utcfromtimestamp(time_dist[i * width + width // 2]))

plt.subplot(2, 1, 1)
for i in range(3):
    plt.plot(time_dist, fsm_B[:, i], color=["r", "g", "b"][i], alpha=0.6)

plt.subplot(2, 1, 2)
for i in range(3):
    plt.plot(
        x,
        kurtosisss[:, i],
        alpha=0.6,
        color=["r", "g", "b"][i],
        label=f"$\mu_4(B_{['x','y','z'][i]})={np.mean(kurtosisss, axis=0)[i]}$",
    )
# plt.plot(
#     x,
#     np.linalg.norm(kurtosisss, axis=1),
#     color="k",
#     label=r"$\mu_4(B)={0}$".format(np.mean(np.linalg.norm(kurtosisss, axis=1), axis=0)),
# )

plt.title("Kurtosis of B field")
plt.ylabel("Kurtosis (3=gaussian)")
plt.xlabel("Time")
plt.legend()
# plt.savefig("src/magSpec/img/kurtosis_B_132100_Tue23Feb.png")
plt.tight_layout()
plt.show()
