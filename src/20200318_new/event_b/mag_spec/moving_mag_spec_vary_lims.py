"""
Last modified:  19/04/21

Moving windows of magnetic spectrum, with variable ion & electron limits.

"""

import os
import numpy as np
import matplotlib.pyplot as plt
import logging as log
import time as timetime
import json
from scipy.optimize import curve_fit
from tqdm import tqdm

path = os.path.dirname(os.path.realpath(__file__))
dirpath = "/".join(path.split("/")[:-1])
print(dirpath)

log.basicConfig(
    filename=f"{path}/mag_spec.log",
    level=log.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)


def extract_grads(k, y, ion, elec, instr):
    def split(X, Y, low, high):
        mask_x = X[(X >= low) & (X <= high)]
        mask_y = Y[(X >= low) & (X <= high)]

        return (np.log10(mask_x), np.log10(mask_y))

    def fit(X, Y):
        grad, pcov = curve_fit(lambda x, m, c: c + m * x, X, Y)
        return grad[0]

    # Iner
    a, b = split(k, y, k[0], ion)
    iner = fit(a, b)

    # Ion
    a, b = split(k, y, ion, elec)
    ion = fit(a, b)

    # Elec
    a, b = split(k, y, elec, instr)
    elec = fit(a, b)

    return np.array([iner, ion, elec])


def lengths(s, number_density, temp_perp, B_field):
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

    if s == "i":
        return rho
    else:
        return p


log.info("Loading data")
big_data = np.load(f"{dirpath}/data/fsm/data.npy")
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
log.info(f"max index: {max_index}")
bin_starts = np.linspace(0, max_index, N, dtype=int)

log.info(f"Ion lim: {ion_lim}\nelectron lim: {electron_lim}")

grads = []
times = []

for bin in tqdm(bin_starts):
    Y = {}

    data = big_data[bin : bin + bin_size, :]
    time = big_time[bin : bin + bin_size]

    # log.info("Comutping FFT over each coord")
    for i in range(3):
        # log.info(f"index {i}")
        B = data[:, i] * 1e-9
        # log.info("Scaling mean")
        B -= B.mean()

        # log.info("Applying Hanning window")
        Hann = np.hanning(len(B)) * B
        # log.info("Calculating FFT")
        Yi = np.fft.fft(Hann)
        # log.info("Calculating Frequencies")
        freq = np.fft.fftfreq(len(B), td)
        # log.info("Obtaining power spectrum")
        Y[["x", "y", "z"][i]] = (np.power(np.abs(Yi), 2) * 1e9 * td)[freq > 0]
    # log.info("Summing components")
    y = np.sum([Y[i] for i in ["x", "y", "z"]], axis=0)
    k = freq[freq > 0] * 2 * np.pi / meanv

    number_density_i = big_number_density_i[(time_i >= time[0]) & (time_i <= time[-1])]
    temp_perp_i = big_temp_perp_i[(time_i >= time[0]) & (time_i <= time[-1])]

    number_density_e = big_number_density_e[(time_e >= time[0]) & (time_e <= time[-1])]
    temp_perp_e = big_temp_perp_e[(time_e >= time[0]) & (time_e <= time[-1])]

    ion_lim = 1.0 / lengths("i", number_density_i, temp_perp_i, data)
    electron_lim = 1.0 / lengths("e", number_density_e, temp_perp_e, data)

    grads.append(extract_grads(k, y, ion_lim, electron_lim, 10))
    times.append(big_time[bin + bin_size // 2])

    plt.loglog(k, y)
    lims = (1e-16, 1)
    plt.vlines(
        [ion_lim, electron_lim, 10],
        np.min(lims[0]),
        np.max(lims[1]),
        ("red", "green", "blue"),
    )
    plt.ylim(lims)
    plt.savefig(f"{path}/anim/{bin}.png")
    plt.clf()

    del y
    del Y
    del freq
    del B
    del Hann
    del Yi
    del data
    del time

grads = np.array(grads)
np.save(f"{path}/grads.npy", grads)
np.save(f"{path}/times.npy", times)

fig, ax = plt.subplots(2, 1, sharex=True)
ax[0].plot(big_time, np.linalg.norm(big_data, axis=1))

for i in range(3):
    ax[1].plot(
        times,
        grads[:, i],
        color=["r", "g", "b"][i],
        label=["inertial", "ion", "electron"][i],
    )
    ax[1].legend()

plt.show()
