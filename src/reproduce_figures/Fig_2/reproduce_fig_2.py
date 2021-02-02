import pyspedas
from pytplot import tplot  # Plotting Library
from pytplot import store_data

# -------- LOAD DATA
trange = ["2017-01-26/08:13:04", "2017-01-26/08:17:04"]
probe = "1"
data_rate = "brst"
# Load FPI data (electron & ion)
mms_fgm = pyspedas.mms.fpi(trange=trange, probe=probe, data_rate=data_rate)
# Load FGM data (B)
# mms_fgm = pyspedas.mms.fgm(trange=trange, probe=probe, data_rate=data_rate)
fgm_all = pyspedas.mms.fgm(
    trange=["2017-01-26/08:13:04", "2017-01-26/08:17:04"],
    probe=["1", "2", "3", "4"],
    data_rate="brst",
)


# -------- MERGE perp & para T into single tplot var
# mms1_des_temppara_brst = electron temp parallel
# mms1_des_tempperp_brst  = electron temp perpendicular
# Merge
store_data(
    "mms1_des_tempparaperp_brst",
    data=["mms1_des_temppara_brst", "mms1_des_tempperp_brst"],
)

# mms1_dis_temppara_brst = ion temp para
# mms1_dis_tempperp_brst = ion temp perp
# Merge
store_data(
    "mms1_dis_tempparaperp_brst",
    data=["mms1_dis_temppara_brst", "mms1_dis_tempperp_brst"],
)

# -------- CURL
# In-built function that uses curlometer technique to
# calculate curl of B (along with some other params)
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

# -------- PLOT
tplot(
    [
        "mms1_fgm_b_gse_brst_l2",
        "jtotal",
        "mms1_dis_bulkv_gse_brst",
        "mms1_des_numberdensity_brst",
        "mms1_des_tempparaperp_brst",
        "mms1_dis_tempparaperp_brst",
        "mms1_dis_energyspectr_omni_brst",
        "mms1_des_energyspectr_omni_brst",
    ]
)
