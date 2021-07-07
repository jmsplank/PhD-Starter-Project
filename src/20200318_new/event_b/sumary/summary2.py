# Summary statistics
import numpy as np
import matplotlib.pyplot as plt
from phdhelper.helpers import os_shortcuts as oss
import os


def lengths(s, number_density, temp_perp, B_field, all=False):
    n = number_density.mean()
    const = 1.32e3 if s == "i" else 5.64e4
    # https://en.wikipedia.org/wiki/Plasma_parameters#Fundamental_plasma_parameters
    omega_p = const * np.sqrt(n)
    p = 2.99792458e8 / omega_p
    p /= 1e3

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

    omega_c = 1.60217662e-19 * BT / (1.6726219e-27 if s == "i" else 9.10938356e-31)
    rho = v / omega_c
    # return limit

    if all:
        return np.array([rho, p])
    else:
        if s == "i":
            return rho
        else:
            return p


data_path = oss.new_path(oss.get_path(__file__, ".."))

shock_time = 1584500550.8

B = np.load(data_path("data/fgm/data.npy"))
time_B = np.load(data_path("data/fgm/time.npy"))

bulkv_i = np.load(data_path("data/fpi/data_bulkv_i.npy"))
time_i = np.load(data_path("data/fpi/time_i.npy"))
# time_e = np.load(data_path("data/fpi/time_e.npy"))

numdens_i = np.load(data_path("data/fpi/data_numberdensity_i.npy"))
numdens_e = np.load(data_path("data/fpi/data_numberdensity_e.npy"))
numdens_e_time = np.load(data_path("data/fpi/time_numberdensity_e.npy"))

tempperp_i = np.load(data_path("data/fpi/data_tempperp_i.npy"))
tempperp_e = np.load(data_path("data/fpi/data_tempperp_e.npy"))
tempperp_e_time = np.load(data_path("data/fpi/time_tempperp_e.npy"))

v = {
    "all": np.linalg.norm(bulkv_i, axis=1).mean(),
    "sw": np.linalg.norm(bulkv_i[time_i >= shock_time], axis=1).mean(),
    "str": np.linalg.norm(bulkv_i[time_i < shock_time], axis=1).mean(),
}

gyrorad_i = {
    "sw": lengths(
        "i",
        numdens_i[time_i >= shock_time],
        tempperp_i[time_i >= shock_time],
        B[time_B >= shock_time],
        all=True,
    )[0],
    "str": lengths(
        "i",
        numdens_i[time_i < shock_time],
        tempperp_i[time_i < shock_time],
        B[time_B < shock_time],
        all=True,
    )[0],
    "all": lengths("i", numdens_i, tempperp_i, B, all=True)[0],
}
gyrorad_e = {
    "sw": lengths(
        "e",
        numdens_e[numdens_e_time >= shock_time],
        tempperp_e[tempperp_e_time >= shock_time],
        B[time_B >= shock_time],
        all=True,
    )[0],
    "str": lengths(
        "e",
        numdens_e[numdens_e_time < shock_time],
        tempperp_e[tempperp_e_time < shock_time],
        B[time_B < shock_time],
        all=True,
    )[0],
    "all": lengths("e", numdens_e, tempperp_e, B, all=True)[0],
}
inert_len_i = {
    "sw": lengths(
        "i",
        numdens_i[time_i >= shock_time],
        tempperp_i[time_i >= shock_time],
        B[time_B >= shock_time],
        all=True,
    )[1],
    "str": lengths(
        "i",
        numdens_i[time_i < shock_time],
        tempperp_i[time_i < shock_time],
        B[time_B < shock_time],
        all=True,
    )[1],
    "all": lengths("i", numdens_i, tempperp_i, B, all=True)[1],
}
inert_len_e = {
    "sw": lengths(
        "e",
        numdens_e[numdens_e_time >= shock_time],
        tempperp_e[tempperp_e_time >= shock_time],
        B[time_B >= shock_time],
        all=True,
    )[1],
    "str": lengths(
        "e",
        numdens_e[numdens_e_time < shock_time],
        tempperp_e[tempperp_e_time < shock_time],
        B[time_B < shock_time],
        all=True,
    )[1],
    "all": lengths("e", numdens_e, tempperp_e, B, all=True)[1],
}

names = [
    "Region",
    r"$v$ $[kms^{-1}]$",
    r"$\rho_i$ $[km]$",
    r"$d_i$ $[km]$",
    r"$\rho_e$ $[km]$",
    r"$d_i$ $[km]$",
]
values = [v, gyrorad_i, inert_len_i, gyrorad_e, inert_len_e]
sep = "\n  | "
table = f"""\\begin{{table}}
\\centering
\\begin{{tabularx}}{{\linewidth}} {{
    {sep.join(['>{§centering§arraybackslash}X']*len(names))} }}
""".replace(
    "§", "\\"
)
table += r"  \hline\hline " + "\n"
table += "  " + " & ".join(names)
table += r" \\" + "\n" + r"  \hline " + "\n"
for k, i in enumerate(["str", "sw", "all"]):
    table += "  "
    table += ["STR & ", "SW & ", "Total & "][k]
    table += r" & ".join(map(lambda k: f"${k[i]:.2f}$", values)) + r" \\" + "\n"
table += r"  \hline" + "\n"
table += r"\end{tabularx}" + "\n"
table += r"\end{table}"
print(table)
# os.system(f"echo {table} | pbcopy")