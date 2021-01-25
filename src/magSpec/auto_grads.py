import pyspedas
import numpy as np
from pytplot import tplot, data_quants
from ..distribution_function.rotate import rot_to_b

trange = ["2016-12-09/09:01:36", "2016-12-09/09:07:00"]  # Interval 1
# trange = ["2016-12-09/09:26:24", "2016-12-09/09:34:58"]  # Interval 2
probe = "1"
# probe = "1"
data_rate = "brst"

pyspedas.mms.fpi(trange, probe, data_rate)
pyspedas.mms.fgm(trange, probe, data_rate)

# tplot(["mms1_fgm_b_gse_brst_l2", "mms1_dis_tempperp_brst"])

print("---->----IONS----<----")
print("Getting number diensity mean")
n_i = data_quants["mms1_dis_numberdensity_brst"].values.mean()
print("Calc ion plasma freq")
i_const = 1.32e3
mu = 1
Z = 1
# https://en.wikipedia.org/wiki/Plasma_parameters#Fundamental_plasma_parameters
omega_pi = i_const * Z * np.sqrt(n_i / mu)
print("Calc ion inert. len")
p_i = 2.99792458e8 / omega_pi
print(p_i)

print("Calc ion gyrorad.")
print("Calc ion gyrofreq.")
v_i = data_quants["mms1_dis_bulkv_gse_brst"].values.mean(axis=0)
B = data_quants["mms1_fgm_b_gse_brst_l2"].values
BT = B[:, 3].mean()
B = B[:, :3].mean(axis=0)
v_i = rot_to_b(B, v_i)
v_i = np.linalg.norm(v_i[:2])
print(v_i)
# B = data_quants["mms1_fgm_b_gse_brst_l2"].values[:, 3].mean() * 1e-9
# print(f"{B/1e-9:0.3f}nT")
# T_i = data_quants["mms1_dis_tempperp_brst"].values.mean() * (
#     1.60217662e-19 / 1.38064852e-23
# )
# print(f"{T_i:0.3f}K")
# rho_i = 1.02e2 * ((mu * T_i) ** 0.5) / (Z * B)
# print(rho_i)
