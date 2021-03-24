from numpy.core.fromnumeric import mean, size
import pyspedas
from pytplot import data_quants
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import correlate, correlation_lags
from phdhelper.helpers import override_mpl
from scipy.optimize import curve_fit
from scipy.ndimage.filters import uniform_filter1d
from phdhelper.helpers.CONSTANTS import c, m_i, q, epsilon_0

override_mpl.override()


def corr_xyz(data, meanv, d_i, td):
    out = np.empty((data.shape[0], 3))

    for i in range(3):
        dta = data[:, i] - data[:, i].mean()
        corr = correlate(dta, dta, mode="full")
        out[:, i] = corr[corr.size // 2 :] / corr[corr.size // 2]

    norm_out = out.mean(axis=1)
    lags = correlation_lags(len(data), len(data), mode="full")
    lags = lags[lags.size // 2 :]
    l = lags * meanv * td / d_i

    return l, norm_out


def get_stop(data):
    # Find 1st 0 crossing
    cross0 = np.argmax(data[1:] * data[:-1] < 0)
    # Find 1st minimum
    minimum = np.argmax(data[1:] - data[:-1] > 0)
    return cross0 if cross0 < minimum else minimum


def exp(x, A, B):
    return B * np.exp(A * x)


def fit(y, x, stop, eqn):
    if eqn.__name__ == "exp":
        fit, err = curve_fit(eqn, x[:stop], y[:stop], p0=(-1.0 / 13, 1))
    else:
        y = np.log(y)
        fit, err = curve_fit(eqn, x[:stop], y[:stop])

    l_c = -1.0 / fit[0]

    if eqn.__name__ == "exp":
        val = eqn(x[:stop], *fit)
        return l_c, val
    else:
        val = eqn(x[:stop], *fit)
        return l_c, np.exp(val)


def grab_t(a, b):
    start = a[0]
    stop = a[-1]

    start = np.argmin(abs(start - b))
    stop = np.argmin(abs(stop - b))
    if stop < len(b) - 1:
        stop += 1
    return [start, stop]


def _d_i(number_density):
    number_density = number_density.mean()
    number_density *= 1e6

    plasma_freq = np.sqrt((number_density * q ** 2) / (m_i * epsilon_0))

    d_i = c / plasma_freq
    d_i /= 1e3

    return d_i


if __name__ == "__main__":
    # trange = ["2020-03-18/02:48:00", "2020-03-18/03:09:00"]
    trange = ["2020-03-18/02:05:00", "2020-03-18/02:44:00"]
    probe = "1"
    data_rate = "brst"
    level = "l2"
    pyspedas.mms.fgm(trange=trange, probe=probe, data_rate=data_rate, level=level)
    pyspedas.mms.fpi(trange=trange, probe=probe, data_rate=data_rate, level="l2")

    v = data_quants["mms1_dis_bulkv_gse_brst"].values
    vt = data_quants["mms1_dis_bulkv_gse_brst"].coords["time"].values

    nd = data_quants["mms1_dis_numberdensity_brst"].values
    ndt = data_quants["mms1_dis_numberdensity_brst"].coords["time"].values

    fgm_str = lambda probe_num: f"mms{probe_num}_fgm_b_gse_brst_l2"
    B = data_quants[fgm_str(probe)].values[:, :3]  # B_x,y,z
    T = data_quants[fgm_str(probe)].coords["time"].values  # Time

    avg_meanv = np.load("src/Event_2/stats/meanv.npy")
    avg_d_i = np.load("src/Event_2/stats/d_i.npy")
    meanv = avg_meanv
    d_i = avg_d_i

    dt = T[1] - T[0]  # s
    print("dt", dt)
    print(len(T))

    # Split into windows
    len_T = len(T)
    every = (20 * avg_d_i) / avg_meanv  # s
    step = np.argmin(abs(T - (T[0] + every)))

    status_str_0 = (
        f"Splitting {len_T} measurements "
        f"every {every:.2f}s into ~{len_T//step} steps (non-overlapping)"
    )
    print(status_str_0)

    times = []
    corr_lens = []
    window = 0
    while window + step <= len_T:
        # print(f"window {window}")
        b = B[window : window + step, :]
        t = T[window : window + step]
        times.append(t[len(t) // 2])

        # meanv = grab_t(t, vt)
        # meanv = np.linalg.norm(v[meanv[0] : meanv[1]], axis=1).mean()
        d_i = grab_t(t, ndt)
        d_i = _d_i(nd[d_i[0] : d_i[1]])

        x, y = corr_xyz(b, meanv, d_i, dt)
        stop = get_stop(y)
        lfit = fit(y, x, stop, exp)
        corr_lens.append(lfit[0])

        print(
            f"window {window:07d} | meanv {meanv:03.2f} | d_i {d_i:03.2f} | corr_len {lfit[0]:02.2f}"
        )

        window += step

    data = {
        "corr_lens": np.array(corr_lens),
        "times": np.array(times),
    }
    np.save("src/quaspara_CS/moving_correlation/mov_corr.npy", data)

    status_str_1 = (
        f"Generated {len(times)} correlation lengths "
        f"with a mean of {np.mean(corr_lens):02.2f} "
        f"meanv: {avg_meanv:.1f} km/s | d_i: {avg_d_i:.1f} km"
    )
    print(status_str_1)

    fig, ax = plt.subplots(2, 1, sharex=True)

    ax[0].plot(T, np.linalg.norm(B, axis=1), label="$|B|$")
    filt = 16384
    ax[0].plot(
        uniform_filter1d(T, size=filt, mode="nearest"),
        uniform_filter1d(np.linalg.norm(B, axis=1), size=filt, mode="nearest"),
        lw=2,
        label=f"{filt} step moving average",
    )
    ax[0].set_ylabel("$|B| \quad [nT]$")
    ax[0].legend()

    ax[1].plot(times, corr_lens, label="correlation length")
    filt = 32
    ax[1].plot(
        uniform_filter1d(times, size=filt, mode="nearest"),
        uniform_filter1d(corr_lens, size=filt, mode="nearest"),
        lw=2,
        label=f"{filt} step moving average",
    )
    ax[1].set_ylabel(f"$l \quad [d_i]$")
    ax[1].legend()

    ax[1].set_xlabel(f"{status_str_0}\n{status_str_1}")

    plt.tight_layout()
    plt.show()
