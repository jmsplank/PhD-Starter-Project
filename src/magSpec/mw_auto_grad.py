from auto_grads import lengths

from datetime import datetime as dt
import os
import asyncio

import matplotlib.pyplot as plt
import numpy as np
import pyspedas

from pytplot import data_quants, tplot
from scipy.optimize import curve_fit

import fsm_magSpec

trange = ["2016-12-09/09:01:36", "2016-12-09/09:07:00"]  # Interval 1
# trange = ["2016-12-09/09:26:24", "2016-12-09/09:34:58"]  # Interval 2
probe = "1"
# probe = "1"
data_rate = "brst"

pyspedas.mms.fpi(trange, probe, data_rate)
pyspedas.mms.fgm(trange, probe, data_rate)


def background(f):
    def wrapped(*args, **kwargs):
        return asyncio.get_event_loop().run_in_executor(None, f, *args, **kwargs)


def load_moving(trange, probe, meanv, width=10e-4, windows=10):
    """Load data as moving window."""

    # Load data from CDF
    pyspedas.mms.fsm(trange=trange, probe=probe, time_clip=True, level="l3")

    # Get data & time from tplot var
    fsm_B = data_quants["mms1_fsm_b_gse_brst_l3"].values
    time_dist = data_quants["mms1_fsm_b_gse_brst_l3"].coords["time"].values
    td = time_dist[1] - time_dist[0]
    # print(td)

    # Interpolate over missing data
    for i in range(3):
        B = fsm_B[:, i] * 10e-09
        finiteMask = np.isfinite(B)
        B = np.interp(time_dist, time_dist[finiteMask], B[finiteMask])
        B -= B.mean()

    tot_len = len(fsm_B)
    # Find number of bins to get specified freq
    N = int(np.floor((2 * np.pi / (td * meanv)) * (1.0 / width)))
    Hann = np.hanning(N)

    # Total number of windows
    out = np.empty((windows, N // 2), dtype=np.float64)

    t = np.fft.fftfreq(N, td)
    t = t[: len(t) // 2]

    # @background
    def fft(i, fsm_B, N, Hann):
        B = fsm_B[i : N + i, :]
        FT = {}
        for j in range(3):
            ft = np.fft.fft(Hann * B[:, j])
            ft = np.power(abs(ft), 2)
            ft *= 1.0 * (1.0 / td)
            FT[["x", "y", "z"][j]] = ft[len(ft) // 2 :]
        y = np.sum([FT[k] for k in ["x", "y", "z"]], axis=0)
        return y

    for a, i in zip(range(windows), np.linspace(0, tot_len - N, windows, dtype=int)):
        out[a, :] = fft(i, fsm_B, N, Hann)

    return t, out


meanv = data_quants["mms1_dis_bulkv_gse_brst"].values.mean(axis=0)
meanv = np.linalg.norm(meanv)
print(f"mean velocity v₀: {meanv}km/s")

k, wins = load_moving(trange=trange, probe=probe, meanv=meanv)

i_limit = np.argmin(abs((1.0 / lengths("i")) - k))
e_limit = np.argmin(abs((1.0 / lengths("e")) - k))
instrum_limit = np.argmin(abs(k - 10))

for i in range(wins.shape[0]):
    plt.subplot(wins.shape[0] // 2, 2, i + 1)
    plt.loglog(k, wins[i, :])
plt.tight_layout()
plt.show()