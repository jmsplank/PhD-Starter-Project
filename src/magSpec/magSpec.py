"""First attempt at creating a magnetic spectra.
SRC:
10.3847/2041-8213/ab21c8
Properties of the Turbulence Associated with Electron-only Magnetic Reconnection in Earth's Magnetosheath
J. E. Stawarz et al.
"""
import math
from datetime import datetime as dt
import time

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pyspedas
from pytplot import data_quants
from scipy.optimize import curve_fit


def shift_with_nan(xs, n):
    """Shift array elements in xs by n indices.
    SRC: https://stackoverflow.com/q/30399534/4465614
    """
    e = np.empty_like(xs)
    if n >= 0:
        e[:n] = np.nan
        e[n:] = xs[:-n]
    else:
        e[n:] = np.nan
        e[:n] = xs[-n:]
    return e


def interp_correction(ref_str, modify_str, time):
    if ref_str != modify_str:
        pyspedas.tinterpol(modify_str, ref_str, newname=modify_str)
    data = data_quants[modify_str].values[:, :3]
    fm = np.ones(len(data[:, 0]), dtype=bool)
    for i in range(3):
        finiteMask = np.isfinite(data[:, i])
        fm = np.logical_and(fm, finiteMask)
    data2 = np.zeros((len(fm), 3))
    for i in range(3):
        data2[:, i] = np.interp(time, time[fm], data[fm, i])
    return data2


def load_data(trange, probe, data_rate, load_fgm=True, load_scm=True):
    if type(probe) == str:
        probe = [probe]
    if load_fgm:
        fgm_str = lambda probe_num: f"mms{probe_num}_fgm_b_gse_brst_l2"
        print("Loading FGM")
        pyspedas.mms.fgm(
            trange=trange, probe=probe, data_rate=data_rate, time_clip=True
        )
        fgm = {}
        fgm["time"] = data_quants[fgm_str(probe[0])].coords["time"].values
        fgm["time_dt"] = np.array([dt.utcfromtimestamp(x) for x in fgm["time"]])
    if load_scm:
        scm_str = lambda probe_num: f"mms{probe_num}_scm_acb_gse_scb_brst_l2"
        print("Loading SCM")
        pyspedas.mms.scm(
            trange=trange, probe=probe, data_rate=data_rate, time_clip=True
        )
        scm = {}
        scm["time"] = data_quants[scm_str(probe[0])].coords["time"].values
        scm["time_dt"] = np.array([dt.utcfromtimestamp(x) for x in scm["time"]])

    for probe_num in probe:
        if load_fgm:
            fgm[f"mms{probe_num}"] = interp_correction(
                fgm_str(probe[0]), fgm_str(probe_num), fgm["time"]
            )
        if load_scm:
            scm[f"mms{probe_num}"] = interp_correction(
                scm_str(probe[0]), scm_str(probe_num), scm["time"]
            )

    return (fgm if load_fgm else None, scm if load_scm else None)


def separate_components(mag_dict):
    keys = [x for x in mag_dict.keys() if "mms" in x]
    outNames = []
    for key in keys:
        for i in range(3):
            name = f"{key}_{['x', 'y', 'z'][i]}"
            mag_dict[name] = mag_dict[key][:, i]
            outNames.append(name)
    return outNames


def do_FFT(data, xyzs, meanv):
    data["freq"] = np.fft.fftfreq(
        len(data[xyzs[0]]), (data["time"][1] - data["time"][0])
    )
    for xyz in xyzs:
        centered = data[xyz] - np.mean(data[xyz])
        centered_hann = np.hanning(len(centered)) * centered
        Y = np.fft.fft(centered_hann)
        Y = abs(Y)[data["freq"] > 0]
        data[f"{xyz}_FFT"] = Y
    data["freq"] = data["freq"][data["freq"] > 0]
    data["freq_taylor"] = 2 * np.pi * data["freq"] / meanv


def sum_components(data, probe):
    for p in probe:
        keys = [x for x in data.keys() if f"mms{p}" in x and "FFT" in x]
        data[f"mms{p}_FFT"] = np.sum(np.array([data[k] for k in keys]), axis=0)
        print(f"mms{p}_FFT shape: {data[f'mms{p}_FFT'].shape}")


def mean_probes(data, probe):
    data["FFT"] = np.mean(np.array([data[f"mms{p}_FFT"] for p in probe]), axis=0)
    print(f"FFT shape: {data['FFT'].shape}")


def plot_FFT(data, slope=None, color="k"):
    plt.subplot(2 if slope is not None else 1, 1, 1)
    plt.loglog(data["freq_taylor"], data["FFT"], color=color)
    plt.xlabel("f [Hz]")
    plt.ylabel("Magnetic spectrum $[nT^2Hz^{-1}]$")

    gen_str = dt.strftime(dt.now(), "%H%M%S_%a%d%b")
    plt.title(f"Plot generated {gen_str}")

    if slope is not None:
        plt.subplot(2, 1, 2)
        plt.loglog(
            data["freq_taylor"], data["FFT"] * data["freq_taylor"] ** slope, color=color
        )
        plt.xlabel("k[km$^{-1}$]")
        plt.ylabel(r"Magnetic Spectrum $\times k^{2.7}$")

    return gen_str


if __name__ == "__main__":
    trange = ["2016-12-09/09:03:00", "2016-12-09/09:10:00"]
    probe = ["1", "2", "3", "4"]
    # probe = "1"
    data_rate = "brst"

    print("Loading data")
    fgm, scm = load_data(trange, probe, data_rate)

    # TODO: Correct for lag

    print("Separating components")
    fgm_xyz = separate_components(fgm)
    scm_xyz = separate_components(scm)

    meanv = np.load("src/magSpec/meanv.npy", allow_pickle=True)

    print("FFT")
    do_FFT(fgm, fgm_xyz, meanv)
    do_FFT(scm, scm_xyz, meanv)

    print("Σ components")
    sum_components(fgm, probe)
    mean_probes(fgm, probe)

    sum_components(scm, probe)
    mean_probes(scm, probe)

    print("Plotting")
    _ = plot_FFT(fgm, slope=2.7)
    sv = plot_FFT(scm, slope=2.7, color="g")

    plt.savefig(f"src/magSpec/magSpec_{sv}.png")
    plt.show()