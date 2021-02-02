import asyncio
import os

# from datetime import datetime as dt
from math import floor

import matplotlib.pyplot as plt
import numpy as np
import pyspedas
from pytplot import data_quants
from scipy.optimize import curve_fit
from scipy.ndimage.filters import uniform_filter1d

import fsm_magSpec
from auto_grads import lengths, grad

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


def load_moving(trange, probe, meanv, width=10e-5, windows=10):
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
    # N = int(np.floor((2 * np.pi / (td * meanv)) * (1.0 / width)))
    N = int(fsm_B.shape[0] * (1.0 / windows))
    Hann = np.hanning(N)

    # Total number of windows

    T = np.fft.fftfreq(N, td)
    t = T[T > 0]
    t = 2 * np.pi * t / meanv

    out = np.empty((windows, len(t)), dtype=np.float64)

    # @background
    def fft(i, fsm_B, N, Hann):
        # B = fsm_B[:, :] * 1e-9
        B = fsm_B[i : N + i, :] * 1e-9
        FT = {}
        for j in range(3):
            BB = B[:, j] - B[:, j].mean()
            ft = np.fft.fft(Hann * BB)
            ft = np.power(abs(ft), 2)
            ft *= 1e9 / (1.0 / td)
            FT[["x", "y", "z"][j]] = ft[T > 0]
        y = np.sum([FT[k] for k in ["x", "y", "z"]], axis=0)
        return y

    for a, i in zip(range(windows), np.linspace(0, tot_len - N, windows, dtype=int)):
        out[a, :] = fft(i, fsm_B, N, Hann)

    return t, out


def fit_lines(k, data, ilim, elim, ins_lim):
    def line(x, m, c):
        return c + m * x

    def grad(f, x, y, lower=None, upper=None):
        sl = slice(lower, upper)
        grad, pcov = curve_fit(f, x[sl], y[sl])
        err_grad = np.sqrt(np.diag(pcov))
        return grad, err_grad, sl

    klog = np.log10(k)
    datalog = np.log10(data)

    g0 = grad(line, klog, datalog, None, ilim)
    g1 = grad(line, klog, datalog, ilim, elim)
    g2 = grad(line, klog, datalog, elim, ins_lim)

    line_x = []
    line_y = []
    for f in [g0, g1, g2]:
        fx = np.array([klog[f[2]][0], klog[f[2]][-1]])
        line_x.append(10 ** fx)
        line_y.append(10 ** np.array(line(fx, f[0][0], f[0][1])))

    out = {
        "gradient": [f[0][0] for f in [g0, g1, g2]],
        "err_gradient": [f[1][0] for f in [g0, g1, g2]],
        "name": ["None -> ilim", "ilim -> elim", "elim -> ins_lim"],
        "line_x": line_x,
        "line_y": line_y,
    }

    return out


meanv = data_quants["mms1_dis_bulkv_gse_brst"].values.mean(axis=0)
meanv = np.linalg.norm(meanv)
print(f"mean velocity v₀: {meanv}km/s")

k, wins = load_moving(trange=trange, probe=probe, meanv=meanv, windows=6)

i_limit = np.argmin(abs((1.0 / lengths("i")) - k))
e_limit = np.argmin(abs((1.0 / lengths("e")) - k))
instrum_limit = np.argmin(abs(k - 10))

for i in range(wins.shape[0]):
    lfit = fit_lines(k, wins[i, :], i_limit, e_limit, instrum_limit)
    plt.subplot(wins.shape[0] // 2, 2, i + 1)
    ydata = wins[i, :] * k ** -lfit["gradient"][1]
    plt.loglog(k, ydata, alpha=0.8, color="k")
    # plt.loglog(k, uniform_filter1d(ydata, int(5000), mode="reflect"), color="k")
    for i in range(3):
        xdata = lfit["line_x"][i]
        plt.loglog(
            xdata,
            lfit["line_y"][i] * xdata ** -lfit["gradient"][1],
            color=["r", "g", "b"][i],
            label=f"{lfit['name'][i]}: {lfit['gradient'][i]:.2f}",
        )
    plt.legend()

plt.tight_layout()
plt.show()