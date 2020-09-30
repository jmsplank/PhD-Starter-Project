import pyspedas
from pytplot import tplot # Plotting Library
from pytplot import store_data

# -------- LOAD DATA
trange = ['2017-01-26/08:13:04', '2017-01-26/08:17:04']
probe = '1'
data_rate = 'brst'
# Load FPI data (electron & ion)
mms_fgm = pyspedas.mms.fpi(trange=trange, probe=probe, data_rate=data_rate)
# Load FGM data (B)
mms_fgm = pyspedas.mms.fgm(trange=trange, probe=probe, data_rate=data_rate)


# -------- MERGE perp & para T into single tplot var
# mms1_des_temppara_brst = electron temp parallel
# mms1_des_tempperp_brst  = electron temp perpendicular
# Merge
store_data('mms1_des_tempparaperp_brst', data=['mms1_des_temppara_brst', 'mms1_des_tempperp_brst'])

# mms1_dis_temppara_brst = ion temp para
# mms1_dis_tempperp_brst = ion temp perp
# Merge
store_data('mms1_dis_tempparaperp_brst', data=['mms1_dis_temppara_brst', 'mms1_dis_tempperp_brst'])

# -------- CURL
