import pyspedas
import numpy as np
from pytplot import tplot
from pytplot import store_data, data_quants


# -------- LOAD DATA
trange = ["2017-01-26/08:14:58", "2017-01-26/08:15:04"]
probe = "1"
data_rate = "brst"
# Load FPI data (electron & ion)
mms_fgm = pyspedas.mms.fpi(
    trange=trange, probe=probe, data_rate=data_rate, time_clip=True
)
# Load FGM data (B)
# mms_fgm = pyspedas.mms.fgm(trange=trange, probe=probe, data_rate=data_rate)
fgm_all = pyspedas.mms.fgm(
    trange=trange, probe=["1", "2", "3", "4"], data_rate=data_rate, time_clip=True
)

# Merge Ion And Electron Velocities
store_data(
    "ion_e_vel_merge", data=["mms1_des_bulkv_gse_brst", "mms1_dis_bulkv_gse_brst"]
)

# Generate Curl of B
curl_vars = pyspedas.mms.mms_curl(
    [
        "mms1_fgm_b_gse_brst_l2",
        "mms2_fgm_b_gse_brst_l2",
        "mms3_fgm_b_gse_brst_l2",
        "mms4_fgm_b_gse_brst_l2",
    ],
    [
        "mms1_fgm_r_gse_brst_l2",
        "mms2_fgm_r_gse_brst_l2",
        "mms3_fgm_r_gse_brst_l2",
        "mms4_fgm_r_gse_brst_l2",
    ],
)

# Generate magnitude of curl of B
divB = data_quants["curlB"].values  # get value of curl B
divB_time = data_quants["curlB"].coords["time"].values  # get time coordinates

divB_magnitude = np.sqrt(np.sum(np.power(divB, 2), axis=1))  # sqrt of sum of squares
store_data(
    "mms_all_curlB_abs", data={"x": divB_time, "y": divB_magnitude}
)  # store avg in tplot var

# Generate Curl B using bulk flow velocities
# equation nq(v_e-v_i)
n = data_quants["mms1_des_numberdensity_brst"].values  # get number density of electrons
q = 1.60217662e-19  # charge on electron
pyspedas.tinterpol(
    "mms1_des_bulkv_gse_brst", "mms1_des_numberdensity_brst"
)  # interpolate times to match n
v_e = data_quants[
    "mms1_des_bulkv_gse_brst-itrp"
].values  # get evectron bulk flow velocity
pyspedas.tinterpol(
    "mms1_dis_bulkv_gse_brst", "mms1_des_numberdensity_brst"
)  # interpolate times to match n
v_i = data_quants["mms1_dis_bulkv_gse_brst-itrp"].values  # get ion bulk flow velocity

# Precalc
parentheses = v_e - v_i  # subtraction inside parentheses
parentheses = np.sqrt(
    np.sum(np.power(parentheses, 2), axis=1)
)  # sqrt of the sum of the squares

J = n * q * parentheses  # form equation
J_times = (
    data_quants["mms1_des_numberdensity_brst"].coords["time"].values
)  # generate time 'x' array
store_data("mms1_fpi_J", data={"x": J_times, "y": J})  # store as tplot var

# panel (f) - J in perp-para directions (using dot and cross products)

# Para
# get magnetic field
mag = data_quants["mms1_fgm_b_gse_brst_l2"].values[
    :, :3
]  # grab magnetic field components
mag_magnitude = np.sqrt(np.sum(np.power(mag, 2), axis=1))  # get the magnitude

mag_scaleToUnit = mag / mag_magnitude[:, None]  # scale mag to array of unit vectors
dotJB = np.array(
    [np.dot(a, b) for a, b in zip(divB, mag_scaleToUnit)]
)  # compute dot product (np.dot has no axis kwarg)
crossJB = np.cross(divB, mag_scaleToUnit)  # compute cross product
crossJB = np.linalg.norm(crossJB, axis=1)  # get magnitude of cross product


tplot(["mms1_fgm_b_gse_brst_l2", "ion_e_vel_merge", "mms_all_curlB_abs", "mms1_fpi_J"])
