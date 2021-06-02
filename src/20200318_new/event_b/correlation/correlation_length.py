"""
Last modified:  19/04/21

Moving windows of magnetic spectrum, with variable ion & electron limits.

"""

import os
from sys import stderr
import numpy as np
import matplotlib.pyplot as plt
import logging as log
import time as timetime
import json
from scipy.optimize import curve_fit
from scipy import interpolate
from tqdm import tqdm
from phdhelper.helpers import override_mpl
import pandas as pd
import subprocess
from datetime import datetime as dt
from scipy.signal import correlate, correlation_lags, resample
from scipy.optimize import curve_fit

override_mpl.override()
override_mpl.cmaps(name="custom_diverging")


path = os.path.dirname(os.path.realpath(__file__))
dirpath = "/".join(path.split("/")[:-1])
print(dirpath)

log.basicConfig(
    filename=f"{path}/mag_spec.log",
    level=log.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)


def exponential_fit(x, A, B):
    return A * np.exp(-x / B)


def lengths(s, number_density, temp_perp, B_field, all=False):
    if s == "i":
        log.info("---->----IONS----<----")
    else:
        log.info("---->----ELEC----<----")

    n = number_density.mean()
    const = 1.32e3 if s == "i" else 5.64e4
    # https://en.wikipedia.org/wiki/Plasma_parameters#Fundamental_plasma_parameters
    omega_p = const * np.sqrt(n)
    p = 2.99792458e8 / omega_p
    p /= 1e3
    log.info(f"Inertial length: {p:.3f}km")

    T = temp_perp
    v = (
        np.sqrt(
            np.mean(T)
            * 2
            * 1.60217662e-19
            / (1.6726219e-27 if s == "i" else 9.10938356e-31)
        )
        / 1e3
    )
    B_scaled = B_field.copy() * 1e-9
    BT = np.linalg.norm(B_scaled, axis=1).mean()
    log.info(f"V: {v:.3f}kms⁻¹")
    omega_c = 1.60217662e-19 * BT / (1.6726219e-27 if s == "i" else 9.10938356e-31)
    rho = v / omega_c
    log.info(f"Gyroradius: {rho:.3f}km")
    log.info("---->----<<>>----<----")
    log.info("")
    # return limit

    log.info(
        f"n: {n} | const: {const} | omega_p: {omega_p} | v: {v} | BT: {BT} | omega_c: {omega_c}"
    )

    if all:
        return np.array([rho, p])
    else:
        if s == "i":
            return rho
        else:
            return p


log.info("Loading data")
big_data = np.load(f"{dirpath}/data/fsm/data.npy")
big_data_mag = big_data.mean(axis=1)
log.info("Loading time")
big_time = np.load(f"{dirpath}/data/fsm/time.npy")
td = big_time[1] - big_time[0]

log.info("Loading temp_perp")
big_temp_perp_e = np.load(f"{dirpath}/data/fpi/data_tempperp_e.npy")
big_temp_perp_i = np.load(f"{dirpath}/data/fpi/data_tempperp_i.npy")
log.info("Loading number_density")
big_number_density_e = np.load(f"{dirpath}/data/fpi/data_numberdensity_e.npy")
big_number_density_i = np.load(f"{dirpath}/data/fpi/data_numberdensity_i.npy")
log.info("Loading electron time")
time_e = np.load(f"{dirpath}/data/fpi/time_e.npy")
log.info("Loading ion time")
time_i = np.load(f"{dirpath}/data/fpi/time_i.npy")
log.info("Loading stats")
with open(f"{dirpath}/data/fpi/stats.json") as f:
    stats = json.load(f)
meanv = stats["mean_v"]["value"]

N = 100  # Number of windows

ion_lim = 1.0 / lengths("i", big_number_density_i, big_temp_perp_i, big_data)
electron_lim = 1.0 / lengths("e", big_number_density_e, big_temp_perp_e, big_data)

min_freq = ion_lim * 5
bin_size = int(1 / (min_freq * td))
print(bin_size)

max_index = len(big_data) - bin_size
bin_starts = np.linspace(0, max_index, N, dtype=int)

corr_lens = []
corr_lens_i = []
corr_times = []

fig, ax = plt.subplots(3, 1, figsize=(6, 8))


for bin in tqdm(bin_starts):
    Y = {}

    data = big_data[bin : bin + bin_size, :]
    time = big_time[bin : bin + bin_size]

    number_density_i = big_number_density_i[(time_i >= time[0]) & (time_i <= time[-1])]
    temp_perp_i = big_temp_perp_i[(time_i >= time[0]) & (time_i <= time[-1])]

    ion_lims = lengths("i", number_density_i, temp_perp_i, data, all=True)[1]

    correlated = np.empty((data.shape[0] * 2 - 1, 3))
    for i in range(3):
        normed_mag = data[:, i] - data[:, i].mean()
        correlated[:, i] = correlate(normed_mag, normed_mag, mode="full")
    correlated = correlated.mean(axis=1)
    correlation_x_axis = correlation_lags(normed_mag.size, normed_mag.size, mode="full")
    correlation_x_axis = meanv * td * correlation_x_axis  # Conversion to length
    correlation_x_axis = (
        correlation_x_axis / ion_lims
    )  # Scale to units of ion inert. len.
    correlated = correlated / correlated[correlation_x_axis == 0]

    horiz_lim = 100

    correlated = correlated[
        (correlation_x_axis >= 0) & (correlation_x_axis <= horiz_lim)
    ]
    correlation_x_axis = correlation_x_axis[
        (correlation_x_axis >= 0) & (correlation_x_axis <= horiz_lim)
    ]

    fit_correlated = correlated[: np.where(correlated <= 0)[0][0]]
    fit_correlated_x = correlation_x_axis[: np.where(correlated <= 0)[0][0]]
    fitted, _ = curve_fit(exponential_fit, fit_correlated_x, fit_correlated, p0=(1, 10))
    fitted_i = np.trapz(fit_correlated, fit_correlated_x)

    corr_lens.append(fitted[1])
    corr_lens_i.append(fitted_i)
    corr_times.append(time[len(time) // 2])

    ax[0].get_shared_x_axes().join(ax[0], ax[1])
    ax[0].plot(big_time, big_data_mag, color="k")
    ax[0].axvspan(time[0], time[-1], alpha=0.2, color="k")

    ax[1].plot(corr_times, corr_lens_i, label="Integrated")
    ax[1].plot(corr_times, corr_lens, label="Fitted")
    ax[1].legend()
    ax[1].set_xlim((big_time[0], big_time[-1]))

    ax[2].plot(
        correlation_x_axis,
        correlated,
    )
    ax[2].plot(
        fit_correlated_x,
        exponential_fit(fit_correlated_x, *fitted),
        label="Exponential fit",
    )
    ax[2].set_xlim((0, horiz_lim))
    ax[2].axhline(0, color="k")
    ax[2].set_ylim((-0.5, 1))
    ax[2].legend()

    ax[0].set_ylabel("|B|")
    ax[1].set_xlabel("Time")
    ax[1].set_ylabel("Correlation length $\lambda_c \quad [d_i]$")
    ax[2].set_xlabel("$l\quad[d_i]$")
    ax[2].set_ylabel("Correlation")

    plt.tight_layout()
    plt.savefig(f"{path}/anim/{bin}.png")

    for i in range(3):
        ax[i].clear()

corr_lens = np.array(corr_lens)
corr_times = np.array(corr_times)
np.save(f"{path}/corr_lens.npy", corr_lens)
np.save(f"{path}/corr_times.npy", corr_times)