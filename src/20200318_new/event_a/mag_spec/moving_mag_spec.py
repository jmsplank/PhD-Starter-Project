import os
import numpy as np
import matplotlib.pyplot as plt
import logging as log
import time as timetime
import json

path = os.path.dirname(os.path.realpath(__file__))
dirpath = "/".join(path.split("/")[:-1])
print(dirpath)

log.basicConfig(
    filename=f"{path}/mag_spec.log",
    level=log.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)


def lengths(s, number_density, temp_perp, B):
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
    B *= 1e-9
    BT = np.linalg.norm(B, axis=1).mean()
    log.info(f"V: {v:.3f}kms⁻¹")
    omega_c = 1.60217662e-19 * BT / (1.6726219e-27 if s == "i" else 9.10938356e-31)
    rho = v / omega_c
    log.info(f"Gyroradius: {rho:.3f}km")

    limit = p if p > rho else rho
    log.info("---->----<<>>----<----")
    log.info("")
    # return limit

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

min_freq = ion_lim / 5
bin_size = int(1 / (min_freq * td))

max_index = len(big_data) - bin_size
bin_starts = np.linspace(0, max_index, N, dtype=int)

log.info(f"Ion lim: {ion_lim}\nelectron lim: {electron_lim}")

for bin in bin_starts:
    Y = {}

    data = big_data[bin : bin + bin_size, :]
    time = big_time[bin : bin + bin_size]

    log.info("Comutping FFT over each coord")
    for i in range(3):
        log.info(f"index {i}")
        B = data[:, i] * 1e-9
        log.info("Scaling mean")
        B -= B.mean()

        log.info("Applying Hanning window")
        Hann = np.hanning(len(B)) * B
        log.info("Calculating FFT")
        Yi = np.fft.fft(Hann)
        log.info("Calculating Frequencies")
        freq = np.fft.fftfreq(len(B), td)
        log.info("Obtaining power spectrum")
        Y[["x", "y", "z"][i]] = (np.power(np.abs(Yi), 2) * 1e9 * td)[freq > 0]
    log.info("Summing components")
    y = np.sum([Y[i] for i in ["x", "y", "z"]], axis=0)
    k = freq[freq > 0] * 2 * np.pi / meanv

    log.info("Calculating ion & electron limits")

    plt.loglog(k, y)
    plt.vlines(ion_lim, 1e-50, 1e-35, "red")
    plt.vlines(electron_lim, 1e-50, 1e-35, "green")
    plt.savefig(f"{dirpath}/mag_spec/anim/{bin}.png")
    plt.ylim((1e-50, 1e-35))
    plt.xlim((1e-5, 2e2))
    plt.clf()

    del y
    del Y
    del freq
    del B
    del Hann
    del Yi
    del data
    del time

# log.info("Saving...")
# np.save(f"{path}/total_y.npy", y)
# np.save(f"{path}/total_freq.npy", freq)

# log.info("Plotting...")
# plt.loglog(freq, y)
# plt.xlabel("Frequency [Hz]")
# plt.ylabel("Magnetic Spectrum [$nT^2Hz^{-1}$]")

# plt.savefig(f"{path}/mag_spec/mag_spec.png")