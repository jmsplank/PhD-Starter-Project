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


def load_from_pyspedas(trange):
    # Load mms data
    print("Loading data.")
    # Load data
    # trange = ["2016-12-09/09:03:00", "2016-12-09/09:04:00"]
    probe = ["1", "2", "3", "4"]
    # probe = "1"
    data_rate = "brst"

    mms_fpi = pyspedas.mms.fpi(
        trange=trange,
        probe=probe,
        # datatype="dis-dist",
        data_rate=data_rate,
        time_clip=True,
    )

    mms_fgm = pyspedas.mms.fgm(
        trange=trange, probe=probe, data_rate=data_rate, time_clip=True
    )

    # mms_fsm = pyspedas.mms.fsm(trange=trange, probe=probe, time_clip=True, level="l3")

    # mms_scm = pyspedas.mms.scm(
    #     trange=trange, probe=probe, data_rate=data_rate, time_clip=True
    # )

    # Get raw distribution data
    get_b_str = lambda probe_num: f"mms{probe_num}_fgm_b_gse_{data_rate}_l2"
    time_dist = data_quants[get_b_str(1)].coords["time"].values

    # get_fsm_str = lambda probe_num: f"mms{probe_num}_scm_acb_gse_scb_brst_l2"

    # Get time of each step
    # fsm_time_dist = data_quants[get_fsm_str(1)].coords["time"].values
    # Convert to datetime

    b_data = {}
    for probe_num in range(1, 5):
        # Interpolating values onto MMS1 time steps
        b_string = get_b_str(probe_num)
        pyspedas.tinterpol(b_string, get_b_str(1), newname=b_string)
        # Get data from tplot vars
        data = data_quants[b_string].values[:, :3]
        # Sum across MMS1-4
        data = data.sum(axis=1)
        # Get rid of NAN's
        finiteMask = np.isfinite(data)
        b_data[f"mms{probe_num}"] = np.interp(
            time_dist, time_dist[finiteMask], data[finiteMask]
        )

    time_dist = np.array([dt.utcfromtimestamp(x) for x in time_dist])

    print(
        f"""Data loaded. {b_data['mms1'].shape[0]} time steps
    Start:  {dt.strftime(time_dist[0], '%H:%M:%S.%f')}
    End:    {dt.strftime(time_dist[-1], '%H:%M:%S.%f')}"""
    )

    print("Summing B from mms1-4")
    temp = np.empty((b_data["mms1"].shape[0], 4))
    shifts = []
    for i in range(1, 5):
        print(f"MMS{i}: Assigning data")
        temp[:, i - 1] = b_data[f"mms{i}"]
    #     if i > 1:
    #         print(f"MMS{i}: Correlating with MMS1")
    #         # See src/shock_normal/timing_analysis.py
    #         corr = np.correlate(
    #             temp[:, i - 1] - temp[:, i - 1].mean(),
    #             temp[:, 0] - temp[:, 0].mean(),
    #             mode="full",
    #         )
    #         shift = np.argmax(corr) - (len(temp[:, i - 1]) - 1)
    #         shifts.append(shift)
    #         # Shift SC data to same time point
    #         print(f"MMS{i}: Shifting {shift} indices")
    #         temp[:, i - 1] = shift_with_nan(temp[:, i - 1], shift)

    avg_B = np.mean(temp, axis=1)
    shifts = np.array(shifts)
    # Generate slice to trim nan's created by aligning SC's
    slice_B = slice(
        max(shifts[shifts > 0]) if len(shifts[shifts > 0]) > 0 else None,
        -1 + min(shifts[shifts < 0]) if len(shifts[shifts < 0]) > 0 else None,
    )
    avg_B = avg_B[slice_B]
    time_dist = time_dist[slice_B]

    # Saving results
    print("Saving results")
    np.save("src/magSpec/avg_B.npy", avg_B)
    np.save("src/magSpec/time_dist", time_dist)

    return (avg_B, time_dist)


def load_scm_from_pyspedas(trange):
    # Load mms data
    print("Loading data.")
    # Load data
    # trange = ["2016-12-09/09:03:00", "2016-12-09/09:04:00"]
    probe = ["1", "2", "3", "4"]
    # probe = "1"
    data_rate = "brst"

    mms_fpi = pyspedas.mms.fpi(
        trange=trange,
        probe=probe,
        # datatype="dis-dist",
        data_rate=data_rate,
        time_clip=True,
    )

    # mms_fgm = pyspedas.mms.fgm(
    #     trange=trange, probe=probe, data_rate=data_rate, time_clip=True
    # )

    # mms_fsm = pyspedas.mms.fsm(trange=trange, probe=probe, time_clip=True, level="l3")

    mms_scm = pyspedas.mms.scm(
        trange=trange, probe=probe, data_rate=data_rate, time_clip=True
    )

    # Get raw distribution data
    # get_b_str = lambda probe_num: f"mms{probe_num}_fgm_b_gse_{data_rate}_l2"
    # time_dist = data_quants[get_b_str(1)].coords["time"].values

    get_fsm_str = lambda probe_num: f"mms{probe_num}_scm_acb_gse_scb_brst_l2"

    # Get time of each step
    fsm_time_dist = data_quants[get_fsm_str(1)].coords["time"].values
    # Convert to datetime

    b_data = {}
    for probe_num in range(1, 5):
        # Interpolating values onto MMS1 time steps
        b_string = get_fsm_str(probe_num)
        pyspedas.tinterpol(b_string, get_fsm_str(1), newname=b_string)
        # Get data from tplot vars
        data = data_quants[b_string].values[:, :3]
        # Sum across MMS1-4
        data = data.sum(axis=1)
        # Get rid of NAN's
        finiteMask = np.isfinite(data)
        b_data[f"mms{probe_num}"] = np.interp(
            fsm_time_dist, fsm_time_dist[finiteMask], data[finiteMask]
        )

    fsm_time_dist = np.array([dt.utcfromtimestamp(x) for x in fsm_time_dist])

    print(
        f"""Data loaded. {b_data['mms1'].shape[0]} time steps
    Start:  {dt.strftime(fsm_time_dist[0], '%H:%M:%S.%f')}
    End:    {dt.strftime(fsm_time_dist[-1], '%H:%M:%S.%f')}"""
    )

    print("Summing B from mms1-4")
    temp = np.empty((b_data["mms1"].shape[0], 4))
    shifts = []
    for i in range(1, 5):
        print(f"MMS{i}: Assigning data")
        temp[:, i - 1] = b_data[f"mms{i}"]
    #     if i > 1:
    #         print(f"MMS{i}: Correlating with MMS1")
    #         # See src/shock_normal/timing_analysis.py
    #         corr = np.correlate(
    #             temp[:, i - 1] - temp[:, i - 1].mean(),
    #             temp[:, 0] - temp[:, 0].mean(),
    #             mode="full",
    #         )
    #         shift = np.argmax(corr) - (len(temp[:, i - 1]) - 1)
    #         shifts.append(shift)
    #         # Shift SC data to same time point
    #         print(f"MMS{i}: Shifting {shift} indices")
    #         temp[:, i - 1] = shift_with_nan(temp[:, i - 1], shift)

    avg_B = np.mean(temp, axis=1)
    shifts = np.array(shifts)
    # Generate slice to trim nan's created by aligning SC's
    slice_B = slice(
        max(shifts[shifts > 0]) if len(shifts[shifts > 0]) > 0 else None,
        -1 + min(shifts[shifts < 0]) if len(shifts[shifts < 0]) > 0 else None,
    )
    fsm_avg_B = avg_B[slice_B]
    fsm_time_dist = fsm_time_dist[slice_B]

    # Saving results
    print("Saving results")
    np.save("src/magSpec/fsm_avg_B.npy", fsm_avg_B)
    np.save("src/magSpec/fsm_time_dist", fsm_time_dist)

    return (fsm_avg_B, fsm_time_dist)


def gen_plot_data(avg_B, time_dist):
    # plt.subplot(2, 1, 2)
    plt.subplot(2, 1, 1)
    res = avg_B - np.mean(avg_B)
    YHann = np.hanning(len(res)) * res
    Y = np.fft.fft(YHann)
    freq = np.fft.fftfreq(len(res), (time_dist[1] - time_dist[0]).total_seconds())

    # Scale using Taylor Hypothesis
    # Get average ion velocity
    # meanv = 0
    # for iprobe in probe:
    #     bulkv = data_quants[f"mms{iprobe}_dis_bulkv_gse_brst"].values
    #     bulkv = np.linalg.norm(bulkv, axis=1)
    #     meanv += bulkv.mean()
    # meanv /= len(probe)
    # np.save("src/magSpec/meanv.npy", meanv)
    meanv = np.load("src/magSpec/meanv.npy", allow_pickle=True)

    ν_0 = meanv  # * np.sin(np.radians(47))
    print(ν_0)
    k = 2 * np.pi * freq / ν_0

    plt.loglog(k[freq > 0], abs(Y)[freq > 0])
    plt.xlabel("f [Hz]")
    plt.ylabel("Magnetic spectrum [nT²Hz⁻¹]")

    plt.subplot(2, 1, 2)

    k = k[freq > 0]
    data = abs(Y)[freq > 0]
    data2 = data * k ** (2.7)
    plt.loglog(k, data2)
    plt.xlabel("k[km$^{-1}$]")
    plt.ylabel(r"Magnetic Spectrum $\times k^{2.7}$")

    ρ_i = 180.0
    ρ_e = 1.0

    x = np.array([1.0 / ρ_i, 1.0 / ρ_e])
    # indX = [find_nearest(k, x[0]), find_nearest(k, x[1])]
    indX = np.searchsorted(k, x, side="left")
    print("Index of ρ_i and ρ_e: ", indX)
    y = np.array([data[indX[0]], data[indX[1]]])
    print(f"x coord: {x[0]}, {x[1]} | y coord: {y[0]}, {y[1]}")
    print(f"Power: {np.diff(np.log(y)) / np.diff(np.log(x))}")
    plt.loglog(x, y * x ** 2.7)


if __name__ == "__main__":
    trange = ["2016-12-09/09:03:00", "2016-12-09/09:10:00"]
    probe = ["1", "2", "3", "4"]
    # probe = "1"
    data_rate = "brst"

    if 1 == 1:
        print("Trying to load saved arrays.")
        try:
            fsm_avg_B = np.load("src/magSpec/fsm_avg_B.npy", allow_pickle=True)
            fsm_time_dist = np.load("src/magSpec/fsm_time_dist.npy", allow_pickle=True)
            print("Loaded SCM arrays")
        except FileNotFoundError as e:
            print("SCM File(s) not found. Generating...")
            fsm_avg_B, fsm_time_dist = load_scm_from_pyspedas(trange)
        try:
            avg_B = np.load("src/magSpex/avg_B.npy", allow_pickle=True)
            time_dist = np.load("src/magSpex/time_dist.npy", allow_pickle=True)
            print("Loaded FGM arrays")
        except FileNotFoundError as e:
            print("FGM File(s) not found. Generating...")
            avg_B, time_dist = load_from_pyspedas(trange)
    else:
        avg_B, time_dist = load_from_pyspedas(trange)
        fsm_avg_B, fsm_time_dist = load_scm_from_pyspedas(trange)

    gen_plot_data(avg_B, time_dist)
    gen_plot_data(fsm_avg_B, fsm_time_dist)

    plt.savefig(f"src/magSpec/magSpec_{dt.strftime(dt.now(), '%H%M%S_%a%m')}.png")
    plt.show()
