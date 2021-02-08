import asyncio
import os

from datetime import datetime as dt
from math import floor

import matplotlib.pyplot as plt
import numpy as np
import pyspedas
from pytplot import data_quants
from scipy.optimize import curve_fit
from tqdm import tqdm

from auto_grads import lengths


def load_moving(trange, probe, meanv, width=-1, windows=10):
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
    if width == -1:
        N = int(fsm_B.shape[0] * (1.0 / windows))
    else:
        N = width

    Hann = np.hanning(N)

    T = np.fft.fftfreq(N, td)
    t = T[T > 0]
    t = 2 * np.pi * t / meanv

    ilim = np.argmin(abs((1.0 / lengths("i")) - t))
    elim = np.argmin(abs((1.0 / lengths("e")) - t))
    ins_lim = np.argmin(abs(t - 10))

    out = np.empty((windows, 3), dtype=np.float64)

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

    def split_and_fit(k, y, ilim, elim, ins_lim):
        g0 = grad(line, k, y, None, ilim)
        g1 = grad(line, k, y, ilim, elim)
        g2 = grad(line, k, y, elim, ins_lim)
        return g0, g1, g2

    for a, i in tqdm(
        zip(range(windows), np.linspace(0, tot_len - N, windows, dtype=int))
    ):
        # print(f"{a+1}/{windows} => {i}:{i+N}/{tot_len}")
        ft = fft(i, fsm_B, N, Hann)
        # print(f"FFT done: length({len(ft)})")
        out[a, :] = split_and_fit(t, ft, ilim, elim, ins_lim)

    t_cent = time_dist[np.linspace(0, tot_len - N, windows, dtype=int) + N // 2]

    return t_cent, out


def line(x, m, c):
    return c + m * x


def grad(f, x, y, lower=None, upper=None, log=False, minimal=True):
    """Calc gradient & general line fit."""
    if not log:
        x = np.log10(x)
        y = np.log10(y)

    sl = slice(lower, upper)
    grad, pcov = curve_fit(f, x[sl], y[sl])
    err_grad = np.sqrt(np.diag(pcov))
    if minimal:
        return grad[0]
    else:
        return grad, err_grad, sl


def fit_lines(k, data, ilim, elim, ins_lim):
    """Returns data about fitted line."""

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


if __name__ == "__main__":
    trange = ["2016-12-09/09:01:36", "2016-12-09/09:07:00"]  # Interval 1
    # trange = ["2016-12-09/09:26:24", "2016-12-09/09:34:58"]  # Interval 2
    probe = "1"
    # probe = "1"
    data_rate = "brst"

    pyspedas.mms.fpi(trange, probe, data_rate)
    pyspedas.mms.fgm(trange, probe, data_rate)

    meanv = data_quants["mms1_dis_bulkv_gse_brst"].values.mean(axis=0)
    meanv = np.linalg.norm(meanv)
    print(f"mean velocity v₀: {meanv}km/s")

    x, out = load_moving(trange, probe, meanv, width=int(1e5), windows=int(100))
    x2 = [dt.utcfromtimestamp(X) for X in x]
    dx = x[1] - x[0]
    print(f"{len(x)} windows with a separation of {dx}s")

    hann = np.hanning(len(x))
    ftt = np.fft.fftfreq(len(x), dx)

    def abs2(x):
        # https://stackoverflow.com/a/49545394
        return x.real ** 2 + x.imag ** 2

    for i in range(3):
        dta = out[:, i] - out[:, i].mean()
        ft = np.fft.rfft(hann * dta, norm="ortho")
        sc = np.fft.irfft(abs2(ft), norm="ortho")
        plt.subplot(3, 1, i + 1)
        plt.plot(
            ftt[ftt >= 0],
            sc[ftt >= 0],
            label=["->ion", "ion->electron", "electron->instrument"][i],
        )
        plt.legend()

    # for i in range(3):
    #     plt.plot(
    #         x,
    #         out[:, i],
    #         label=["->ion", "ion->electron", "electron->instrument"][i],
    #         color=["r", "g", "b"][i],
    #     )
    # plt.legend()
    plt.show()
