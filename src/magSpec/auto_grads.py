import pyspedas
import numpy as np
from pytplot import tplot, data_quants
from phdhelper.math import transforms
import fsm_magSpec
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

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

    v = data_quants[f"mms1_d{s}s_bulkv_gse_brst"].values.mean(axis=0)
    B = data_quants["mms1_fgm_b_gse_brst_l2"].values * 1e-9
    # BT = B[:, 3].mean()
    B = B[:, :3].mean(axis=0)
    v = transforms.rot_to_b(B, v)
    v = np.linalg.norm(v[:2])
    omega_c = 1.60217662e-19 * B[2] / (1.6726219e-27 if s == "i" else 9.10938356e-31)
    rho = v / omega_c
    print(f"Gyroradius: {rho:.3f}km")

    limit = p if p > rho else rho
    print("---->----<<>>----<----")
    return limit


k, y = fsm_magSpec.load_data(trange, probe)
i_limit = np.argmin(abs((1.0 / lengths("i")) - k))
e_limit = np.argmin(abs((1.0 / lengths("e")) - k))
print(i_limit, e_limit)

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
g2 = grad(e_limit, None)

print(g0, g1, g2, sep="\n")

ax1, ax2 = fsm_magSpec.plot(
    k, y, vlines=np.array([1.0 / lengths("i"), 1.0 / lengths("e")]).flatten()
)


def plotfitted(var, color):
    mainLine = var[0]
    minLine = var[0] - np.array([0, var[1][1]])
    maxLine = var[0] + np.array([0, var[1][1]])

    def gen_xy(grad):
        x = log_k[var[2]][[0, -1]]
        y = line(x, grad[0], grad[1])
        x = 10 ** x
        y = 10 ** y
        return x, y

    mainLine = gen_xy(mainLine)
    minLine = gen_xy(minLine)
    maxLine = gen_xy(maxLine)

    ax1.plot(mainLine[0], mainLine[1], color=color)
    ax2.plot(mainLine[0], mainLine[1] * mainLine[0] ** 2.7, color=color)

    ax1.fill_between(mainLine[0], minLine[1], maxLine[1], color=color, alpha=0.2)
    ax2.fill_between(
        mainLine[0],
        minLine[1] * mainLine[0] ** 2.7,
        maxLine[1] * mainLine[0] ** 2.7,
        color=color,
        alpha=0.2,
    )


plotfitted(g0, "orange")
plotfitted(g1, "red")
plotfitted(g2, "skyblue")


plt.show()
