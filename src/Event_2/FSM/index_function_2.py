import pyspedas
from pytplot import data_quants
import numpy as np
import matplotlib.pyplot as plt
from phdhelper.helpers import override_mpl
from datetime import datetime as dt
from matplotlib.dates import DateFormatter


override_mpl.override()

trange = ["2020-03-18/02:48:00", "2020-03-18/03:08:00"]
pyspedas.mms.fsm(trange=trange, probe="1", data_rate="brst", level="l3")

B = data_quants["mms1_fsm_b_gse_brst_l3"].values
T = data_quants["mms1_fsm_b_gse_brst_l3"].coords["time"].values

t = [dt.utcfromtimestamp(i) for i in T]

fig, ax = plt.subplots(1, 1)
ax.plot(t, B)

# fmt = DateFormatter("%H:%M:%S")
# ax.set_major_formatter(fmt)
# fig.autofmt_xdate()


plt.show()