from datetime import datetime as dt
from scipy.optimize import curve_fit
import pyspedas
from pytplot import data_quants
import numpy as np
from scipy.signal import correlate, correlation_lags
import matplotlib.pyplot as plt


def load_interval(trange, probe):
    data_rate = "brst"

    ########### FGM #############
    fgm_str = lambda probe_num: f"mms{probe_num}_fgm_b_gse_brst_l2"
    pyspedas.mms.fgm(trange=trange, probe=probe, data_rate="brst")
    time_dist = data_quants[fgm_str(probe)].coords["time"].values
    timeDelta = time_dist[1] - time_dist[0]

    data = data_quants[fgm_str(probe)].values[:, :3]
    return timeDelta, data


def exp(x, A, B):
    return B * np.exp(A * x)


def line(x, m, c):
    return c + m * x


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


if __name__ == "__main__":
    # trange = ["2016-12-09/09:01:36", "2016-12-09/09:07:00"]  # Interval 1
    trange = ["2016-12-09/09:26:24", "2016-12-09/09:34:58"]  # Interval 2
    probe = "1"

    summary = {}

    td, data = load_interval(trange, probe)
    meanv = 231
    d_i = 50
    x, y = corr_xyz(data, meanv, d_i, td)
    stop = get_stop(y)
    lfit = fit(y, x, stop, exp)
    summary["I1F"] = lfit[0]
    summary["I1I"] = np.trapz(y[:stop], x[:stop])

    plt.plot(
        x,
        y,
        color="skyblue",
        label=r"2: $\int R \equiv \lambda_c = {:.2f}$".format(summary["I1I"]),
    )
    plt.plot(
        x[:stop],
        lfit[1],
        ls="--",
        color="skyblue",
        label=r"2: $R\propto\exp(-l/\lambda_c) \Rightarrow \lambda_c = {:.2f}$".format(
            summary["I1F"]
        ),
    )

    trange = ["2016-12-09/09:01:36", "2016-12-09/09:07:00"]  # Interval 1
    td, data = load_interval(trange, probe)
    meanv = 242
    d_i = 50
    x, y = corr_xyz(data, meanv, d_i, td)
    stop = get_stop(y)
    lfit = fit(y, x, stop, exp)
    summary["I2F"] = lfit[0]
    summary["I2I"] = np.trapz(y[:stop], x[:stop])

    plt.plot(
        x,
        y,
        color="green",
        label=r"1:$\int R \equiv \lambda_c = {:.2f}$".format(summary["I2I"]),
    )
    plt.plot(
        x[:stop],
        lfit[1],
        ls="--",
        color="green",
        label=r"1: $R\propto\exp(-l/\lambda_c) \Rightarrow \lambda_c = {:.2f}$".format(
            summary["I2F"]
        ),
    )

    print(summary)

    plt.hlines(0, 0, 80)
    plt.xlim((0, 80))
    plt.ylim((-0.1, 1))
    plt.legend()
    plt.grid()
    plt.tight_layout()
    # plt.show()
    tstr = dt.strftime(dt.now(), "%H%M%S_%a%d%b")
    plt.savefig(f"src/magSpec/img/correlationLength_{tstr}.png")
    # plt.show()
