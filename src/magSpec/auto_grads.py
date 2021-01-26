from datetime import datetime as dt
import os

import matplotlib.pyplot as plt
import numpy as np
import pyspedas
from phdhelper.math import transforms
from pytplot import data_quants, tplot
from scipy.optimize import curve_fit

import fsm_magSpec

trange = ["2016-12-09/09:01:36", "2016-12-09/09:07:00"]  # Interval 1
# trange = ["2016-12-09/09:26:24", "2016-12-09/09:34:58"]  # Interval 2
probe = "1"
# probe = "1"
data_rate = "brst"

pyspedas.mms.fpi(trange, probe, data_rate)
pyspedas.mms.fgm(trange, probe, data_rate)

# tplot(["mms1_fgm_b_gse_brst_l2", "mms1_dis_tempperp_brst"])
def lengths(s="i"):
    if s == "i":
        print("---->----IONS----<----")
    else:
        print("---->----ELEC----<----")

    n = data_quants[f"mms1_d{s}s_numberdensity_brst"].values.mean()
    const = 1.32e3 if s == "i" else 5.64e4
    # https://en.wikipedia.org/wiki/Plasma_parameters#Fundamental_plasma_parameters
    omega_p = const * np.sqrt(n)
    p = 2.99792458e8 / omega_p
    p /= 1e3
    print(f"Inertial length: {p:.3f}km")

    v = data_quants[f"mms1_d{s}s_bulkv_gse_brst"].values
    B = data_quants["mms1_fgm_b_gse_brst_l2"].values * 1e-9
    BT = B[:, 3].mean()
    BB = B[:, :3].mean(axis=0)
    Bnorm = BB / np.linalg.norm(BB)
    v2 = np.mean(np.linalg.norm(v, axis=1)) ** 2 - np.mean(np.dot(v, Bnorm)) ** 2
    # print(f"B: {BT*1e9:.3f}nT")
    # B = B[:, :3].mean(axis=0)
    # v = transforms.rot_to_b(B, v)
    # print(v.shape)
    # v = np.linalg.norm(v[:2, :], axis=1)
    # print(v.shape)
    # v = np.linalg.norm(v)
    v = np.sqrt(v2)
    print(f"V: {v:.3f}kms⁻¹")
    omega_c = 1.60217662e-19 * BT / (1.6726219e-27 if s == "i" else 9.10938356e-31)
    rho = v / omega_c
    print(f"Gyroradius: {rho:.3f}km")

    limit = p if p > rho else rho
    print("---->----<<>>----<----")
    print("")
    return limit


k, y = fsm_magSpec.load_data(trange, probe)
i_limit = np.argmin(abs((1.0 / lengths("i")) - k))
e_limit = np.argmin(abs((1.0 / lengths("e")) - k))
instrum_limit = np.argmin(abs(k - 10))

log_k = np.log10(k)
log_y = np.log10(y)


def line(x, m, c):
    return c + m * x


def grad(lower=None, upper=None):
    sl = slice(lower, upper)
    grad, pcov = curve_fit(line, log_k[sl], log_y[sl])
    err_grad = np.sqrt(np.diag(pcov))
    return grad, err_grad, sl


g0 = grad(None, i_limit)
g1 = grad(i_limit, e_limit)
g2 = grad(e_limit, instrum_limit)

# print(g0, g1, g2, sep="\n")

ax1, ax2 = fsm_magSpec.plot(
    k, y, vlines=np.array([1.0 / lengths("i"), 1.0 / lengths("e"), 10]).flatten()
)


def plotci(var, color):
    main = var[0]
    errs = var[1]
    errs = np.column_stack([errs, -errs])
    x = log_k[var[2]]
    y = np.empty((x.shape[0], 4))
    count = 0
    for i in range(2):
        for j in range(2):
            y[:, count] = line(x, main[0] + errs[0, i], main[1] + errs[1, j])
            count += 1
    neg = y.min(axis=1)
    pos = y.max(axis=1)
    main = line(x, main[0], main[1])

    def deloggify(a):
        return 10 ** a

    x = deloggify(x)
    neg = deloggify(neg)
    pos = deloggify(pos)
    main = deloggify(main)

    ax1.plot(x, main, color=color, label=f"{var[0][0]:.1f}")
    ax1.fill_between(x, neg, pos, color=color, alpha=0.2)
    ax1.legend()

    def scale(a, b, grad):
        return a * b ** grad

    neg = scale(neg, x, 2.7)
    pos = scale(pos, x, 2.7)
    main = scale(main, x, 2.7)

    ax2.plot(x, main, color=color)
    ax2.fill_between(x, neg, pos, color=color, alpha=0.2)


plotci(g0, "orange")
plotci(g1, "red")
plotci(g2, "skyblue")

for l, g in zip(["orange", "red", "blue"], [g0, g1, g2]):
    print(f"Gradient of {l} line: {g[0][0]:.2f}±{g[1][0]:.2E}")


dstring = dt.strftime(dt.now(), "%H%M%S_%a%d%b")
ax1.set_title(f"Plot generated: {dstring}")
fname = f"src/magSpec/img/autoSlopes_{dstring}.png"
plt.savefig(fname)
os.system(f"xdg-open {fname}")
keep = input("Keep image? (y/n): ")
if keep[0].lower() != "y":
    os.system(f"mv {fname} {os.path.split(fname)[0]+'/old/'+os.path.split(fname)[1]}")
