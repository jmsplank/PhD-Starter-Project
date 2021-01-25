import time
from datetime import datetime as dt

import matplotlib.pyplot as plt
import numpy as np
import pyspedas
from pytplot import data_quants


def load_data(trange, probe):
    pyspedas.mms.fsm(trange=trange, probe=probe, time_clip=True, level="l3")

    fsm_B = data_quants["mms1_fsm_b_gse_brst_l3"].values
    time_dist = data_quants["mms1_fsm_b_gse_brst_l3"].coords["time"].values
    td = time_dist[1] - time_dist[0]

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
        Y[["x", "y", "z"][i]] = (np.power(abs(Yi), 2) * 10e9 / (1.0 / td))[freq > 0]

    y = np.sum([Y[i] for i in ["x", "y", "z"]], axis=0)
    return k, y


def plot(k, y, vlines=None, slope=2.7):
    ax1 = plt.subplot(2, 1, 1)
    ax1.loglog(k, y)
    ax1.set_xlabel("f [Hz]")
    ax1.set_ylabel("Magnetic spectrum [nT²Hz⁻¹]")
    if vlines is not None:
        ymin = min(y)
        ymax = max(y)
        ax1.vlines(vlines, ymin, ymax)

    ax2 = plt.subplot(2, 1, 2)
    yk = y * k ** slope
    ax2.loglog(k, yk)
    ax2.set_xlabel("k[km$^{-1}$]")
    ax2.set_ylabel(r"Magnetic Spectrum $\times k^{2.7}$")
    if vlines is not None:
        ymin = min(yk)
        ymax = max(yk)
        ax2.vlines(vlines, ymin, ymax)
    return ax1, ax2


if __name__ == "__main__":
    trange = ["2016-12-09/09:01:36", "2016-12-09/09:07:00"]  # Interval 1
    # trange = ["2016-12-09/09:26:24", "2016-12-09/09:34:58"]  # Interval 2
    probe = "1"
    # probe = "1"
    data_rate = "brst"
    k, y = load_data(trange, probe)
    plot(k, y)

    plt.title(f"Generated: {gen_fmt}")
    plt.savefig(f"src/magSpec/fsm_magSpec_{gen_fmt}.png")
    plt.show()