import matplotlib.pyplot as plt
import numpy as np
import pyspedas
from pytplot import data_quants

# Load data
trange = ["2015-10-07/11:44:15", "2015-10-07/11:45:10"]
# probe = ["1", "2", "3", "4"]
probe = '4'
data_rate = "brst"
mms_fgm = pyspedas.mms.fgm(trange=trange,
                           probe=probe,
                           data_rate=data_rate,
                           time_clip=True)

mms_fpi = pyspedas.mms.fpi(trange=trange,
                           probe=probe,
                           data_rate=data_rate,
                           time_clip=True)

# Interpolate b & r to match mms1_b
# for i in range(1, 5):
#     pyspedas.tinterpol(f"mms{i}_fgm_b_gse_brst_l2", "mms1_fgm_b_gse_brst_l2")
# pyspedas.tinterpol(f"mms{i}_fgm_r_gse_brst_l2", "mms1_fgm_b_gse_brst_l2")

pyspedas.tinterpol("mms4_dis_bulkv_gse_brst", "mms4_fgm_b_gse_brst_l2")

# Using MMS4 for analysis
data_b = data_quants["mms4_fgm_b_gse_brst_l2"].values[:, :]
time = data_quants["mms4_fgm_b_gse_brst_l2"].coords['time'].values

data_iv = data_quants["mms4_dis_bulkv_gse_brst-itrp"].values[:, :]

IDEAL = (0.88, 0.46, -0.11)

INDEX = 1000
for i in np.linspace(500, 2750, 10, dtype=int):
    upstream_b = np.mean(data_b[:i, :3], axis=0)
    downstream_b = np.mean(data_b[-i:, :3], axis=0)

    upstream_v = np.nanmean(data_iv[:i, :3], axis=0)
    downstream_v = np.nanmean(data_iv[-i:, :3], axis=0)

    delta_v_ud = downstream_v - upstream_v
    delta_B_ud = downstream_b - upstream_b

    # n= ((Bd X Bu) X DB)/(|(Bd X Bu) X Db|)
    cross_B_du = np.cross(downstream_b, upstream_b)
    cross_B_DB = np.cross(cross_B_du, delta_B_ud)
    n_MC = cross_B_DB / np.linalg.norm(cross_B_DB) * -1.
    resid_MC = np.sum(n_MC - IDEAL)

    # n_MX1
    cross_BuDV = np.cross(upstream_b, delta_v_ud)
    n_MX1 = np.cross(cross_BuDV, delta_B_ud) / np.linalg.norm(
        np.cross(cross_BuDV, delta_B_ud))
    resid_MX1 = np.sum(n_MX1 - IDEAL)

    # n_MX2
    cross_BdDV = np.cross(downstream_b, delta_v_ud)
    n_MX2 = np.cross(cross_BdDV, delta_B_ud) / np.linalg.norm(
        np.cross(cross_BdDV, delta_B_ud))
    resid_MX2 = np.sum(n_MX2 - IDEAL)

    # n_MX3
    cross_dBdV = np.cross(delta_B_ud, delta_v_ud)
    n_MX3 = np.cross(cross_dBdV, delta_B_ud) / np.linalg.norm(
        np.cross(cross_dBdV, delta_B_ud))
    resid_MX3 = np.linalg.norm(n_MX3 - IDEAL)

    # print(
    #     f'''|11:44:15+{time[i]-time[0]:0.1f}s | 11:45:10-{time[-1]-time[-i]:0.1f}s | {resid_MC:0.2f} | {resid_MX1:0.2f} | {resid_MX2:0.2f} | {resid_MX3:0.2f} |'''
    # )
    print(f'''Using first and last {i} indices to average:
    Corresponds to 11:44:15+{time[i]-time[0]:0.1f}s and 11:45:10-{time[-1]-time[-i]:0.1f}s
    n_MC:            ({' '.join([f'{i:0.3f}' for i in n_MC])})
        Residual:    {resid_MC:0.2f}
    n_MX1:           ({' '.join([f'{i:0.3f}' for i in n_MX1])})
        Residual:    {resid_MX1:0.2f}
    n_MX2:           ({' '.join([f'{i:0.3f}' for i in n_MX2])})
        Residual:    {resid_MX2:0.2f}
    n_MX3:           ({' '.join([f'{i:0.3f}' for i in n_MX3])})
        Residual:    {resid_MX3:0.2f}
''')
