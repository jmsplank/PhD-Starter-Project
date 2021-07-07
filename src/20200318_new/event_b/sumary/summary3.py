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
from phdhelper.helpers.COLOURS import red, green, blue, mandarin
import pandas as pd
import subprocess
from datetime import datetime as dt
import matplotlib

override_mpl.override()
override_mpl.cmaps(name="custom_diverging")
matplotlib.rc("font", family="eb garamond")
matplotlib.rc("text", usetex=True)
matplotlib.rc(
    "text.latex",
    preamble=r"""\usepackage[T1]{fontenc}
\usepackage{ebgaramond}
\usepackage{bm}""",
)
matplotlib.rcParams.update({"font.size": 12})


path = os.path.dirname(os.path.realpath(__file__))
path = "/".join(path.split("/")[:-1]) + "/mag_spec"
dirpath = "/".join(path.split("/")[:-1])
print(dirpath)

log.basicConfig(
    filename=f"{path}/summary3.log",
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
knots = []
slope_lims = []
slope_lims_other = []
slope_interp = []
spectra = []
fsm = []


Y = {}
bin_width = bin_starts[1] - bin_starts[0]
shock_time = 1584500550.8
shock_index = np.argmin(np.abs(big_time - shock_time))
data = big_data[shock_index : shock_index + bin_width]
time = big_time[shock_index : shock_index + bin_width]
print(time[-1] - time[0])
print(len(time))

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

ion_lims = 1.0 / lengths("i", number_density_i, temp_perp_i, data, all=True)
electron_lims = 1.0 / lengths("e", number_density_e, temp_perp_e, data, all=True)

ion_lim = ion_lims[0]
electron_lim = electron_lims[1]

instrument_mask = k < 10
kk = np.log10(k[instrument_mask])
yy = np.log10(y[instrument_mask])

f = interpolate.interp1d(kk, yy)
xx = np.log10(np.logspace(kk[0], kk[-1], num=1000))
yy = f(xx)

x_interp = np.linspace(
    np.log10(4e-4),
    np.log10(1),
    32,
)

# spectra.append(10 ** f(x_interp))

r_data = {"x": xx, "y": yy}
r_df = pd.DataFrame(r_data)
r_df.to_csv(f"{path}/raw_r.csv")

devnull = open(os.devnull, "w")
subprocess.call(f"{path}/mars.r", stdout=devnull, stderr=devnull)
r_out = pd.read_csv(f"{path}/mars.csv")
YY = np.array(r_out.y)
slopes_all = np.gradient(YY, abs(xx[0] - xx[1]))

slopes, slope_index, slope_counts = np.unique(
    np.round(slopes_all, 2),
    return_index=True,
    return_counts=True,
)
slopes_all = np.round(slopes_all, 2)
slope_counts = slope_counts > 10
slopes = slopes[slope_counts]
slope_index = slope_index[slope_counts]
slope_k = 10 ** xx[slope_index]

f = interpolate.interp1d(xx, slopes_all, fill_value="extrapolate")
interp_slope = f(x_interp)
slope_interp.append(interp_slope)

slope_lims.append(np.log10(np.array([ion_lim, electron_lim])))
slope_lims_other.append(np.log10(np.array([ion_lims[1], electron_lims[0]])))

ks = np.histogram(
    np.log10(slope_k),
    x_interp,
)[0]
ks = ks.astype("float")
ks[ks == 0] = np.nan
knots.append(ks)

plt.loglog(k, y, alpha=1, color="k")
plt.loglog(10 ** xx, 10 ** YY, color=blue, ls="--", lw=2)
lims = (1e-16, 1e-3)
plt.vlines(slope_k, lims[0], lims[1], mandarin, alpha=1)
plt.vlines(
    [ion_lim, electron_lim],
    np.min(lims[0]),
    np.max(lims[1]),
    (red, green),
)
plt.axvspan(10, max(k), color="k", alpha=0.2)
plt.ylabel("Magnetic Spectrum [$nT^2Hz^{-1}$]")
plt.xlabel("$k$ [$km^{-1}$]")
plt.ylim(lims)
plt.xlim(min(k), max(k))
plt.grid(False)
plt.tight_layout()
plt.show()


# grads = np.array(grads)
# knots = np.array(knots)
# slope_lims = np.array(slope_lims)
# slope_lims_other = np.array(slope_lims_other)
# slope_interp = np.array(slope_interp)
# fsm = np.array(fsm)
# np.save(f"{path}/grads.npy", grads)
# np.save(f"{path}/times.npy", times)
# np.save(f"{path}/knots.npy", np.array(knots))
# np.save(f"{path}/slope_interp.npy", np.array(slope_interp))
# np.save(f"{path}/spectra.npy", np.array(spectra))
# np.save(f"{path}/fsm_sampled_100.npy", fsm)

# fig, ax = plt.subplots(3, 2, gridspec_kw={"width_ratios": [98, 3]})
# ax[0, 0].plot(big_time, np.linalg.norm(big_data, axis=1))

# for i in range(3):
#     ax[1, 0].plot(
#         times,
#         grads[:, i],
#         label=["inertial", "ion_rho", "electron_p"][i],
#     )
#     ax[1, 0].legend()

# im = ax[2, 0].imshow(
#     slope_interp.T,
#     extent=(times[0], times[-1], np.log10(4e-4), np.log10(1)),
#     origin="lower",
#     aspect="auto",
#     vmin=-5,
#     vmax=1,
#     cmap="custom_diverging",
# )
# fig.colorbar(im, cax=ax[2, 1])

# ax[2, 0].imshow(
#     knots.T,
#     extent=(times[0], times[-1], np.log10(4e-4), np.log10(1)),
#     origin="lower",
#     aspect="auto",
# )

# slope_lims[slope_lims == 0] = np.nan
# slope_lims_other[slope_lims_other == 0] = np.nan
# ax[2, 0].plot(times, slope_lims)
# ax[2, 0].plot(times, slope_lims_other)

# for i in range(2):
#     ax[i, 1].axis("off")
#     ax[i, 0].tick_params(
#         axis="x",
#         which="both",
#         bottom="on",
#         top="on",
#         labelbottom=False,
#     )


# ax[0, 0].set_ylabel("$|B|$")
# ax[1, 0].set_ylabel("Slope")
# ax[2, 0].set_ylabel("log k")
# ax[2, 1].set_ylabel("Slope")
# ax[2, 0].set_xlabel("Time")

# for i in range(3):
#     ax[i, 0].set_xticks(big_time[:: len(big_time) // 7])
#     ax[i, 0].set_xticklabels(
#         [
#             dt.strftime(dt.utcfromtimestamp(a), "%H:%M:%S")
#             for a in big_time[:: len(big_time) // 7]
#         ]
#     )

# ax[0, 0].get_shared_x_axes().join(*[ax[i, 0] for i in range(3)])

# plt.tight_layout(pad=0.1)
# plt.show()
