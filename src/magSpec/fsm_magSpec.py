import time
from datetime import datetime as dt

import matplotlib.pyplot as plt
import numpy as np
import pyspedas
from pytplot import data_quants

trange = ["2016-12-09/09:01:36", "2016-12-09/09:07:00"]  # Interval 1
# trange = ["2016-12-09/09:26:24", "2016-12-09/09:34:58"]  # Interval 2
probe = "1"
# probe = "1"
data_rate = "brst"

mms_fsm = pyspedas.mms.fsm(trange=trange, probe=probe, time_clip=True, level="l3")

fsm_B = data_quants["mms1_fsm_b_gse_brst_l3"].values
time_dist = data_quants["mms1_fsm_b_gse_brst_l3"].coords["time"].values


# Correct for missing data
Y = {}
for i in range(3):
    B = fsm_B[:, i] * 10e-9
    finiteMask = np.isfinite(B)
    print((1 - finiteMask).sum())
    B = np.interp(time_dist, time_dist[finiteMask], B[finiteMask])
    B -= B.mean()
    Hann = np.hanning(len(B)) * B
    Yi = np.fft.fft(Hann)
    freq = np.fft.fftfreq(len(B), time_dist[1] - time_dist[0])
    meanv = np.load("src/magSpec/meanv.npy", allow_pickle=True)
    k = 2 * np.pi * freq / meanv
    k = k[freq > 0]
    Y[["x", "y", "z"][i]] = (np.power(abs(Yi), 2) * 10e9 / 125)[freq > 0]

y = np.sum([Y[i] for i in ["x", "y", "z"]], axis=0)

plt.subplot(2, 1, 1)
plt.loglog(k, y)
plt.xlabel("f [Hz]")
plt.ylabel("Magnetic spectrum [nT²Hz⁻¹]")

plt.subplot(2, 1, 2)
plt.loglog(k, y * k ** (2.7))
plt.xlabel("k[km$^{-1}$]")
plt.ylabel(r"Magnetic Spectrum $\times k^{2.7}$")

gen_fmt = dt.strftime(dt.now(), "%H%M%S_%a%d%b")
plt.subplot(2, 1, 1)
plt.title(f"Generated: {gen_fmt}")
plt.savefig(f"src/magSpec/fsm_magSpec_{gen_fmt}.png")
plt.show()
