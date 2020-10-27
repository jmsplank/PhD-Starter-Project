import matplotlib.pyplot as plt
import numpy as np
import pybowshock as pybs
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

positions = data_quants["mms4_fgm_r_gse_brst_l2"].values

RE = 6371
positions = (positions / RE)[0, :3]
print(positions)

IDEAL = (0.88, 0.46, -0.11)

for model in pybs.model_names():
    if 'BS:' in model:
        n_sh = pybs.bs_normal_at_surf_GSE(positions, 160, model)
        resid = np.linalg.norm(n_sh - IDEAL)
        print(f"|Model ({model})| {n_sh[0]:0.3f} | {n_sh[1]:0.3f} | {n_sh[2]:0.3f} | {resid:0.3f}|")
