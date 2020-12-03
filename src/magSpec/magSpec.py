"""First attempt at creating a magnetic spectra.
SRC:
10.3847/2041-8213/ab21c8
Properties of the Turbulence Associated with Electron-only Magnetic Reconnection in Earth's Magnetosheath
J. E. Stawarz et al.
"""
from datetime import datetime as dt

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pyspedas
from pytplot import data_quants


def shift_with_nan(xs, n):
    """Shift array elements in xs by n indices.
    SRC: https://stackoverflow.com/q/30399534/4465614
    """
    e = np.empty_like(xs)
    if n >= 0:
        e[:n] = np.nan
        e[n:] = xs[:-n]
    else:
        e[n:] = np.nan
        e[:n] = xs[-n:]
    return e


# Load mms data
print("Loading data.")
# Load data
trange = ["2016-12-09/09:03:00", "2016-12-09/09:10:00"]
probe = ["1", "2", "3", "4"]
# probe = "1"
data_rate = "brst"

# mms_fpi = pyspedas.mms.fpi(
#     trange=trange,
#     probe=probe,
#     datatype="dis-dist",
#     data_rate=data_rate,
#     time_clip=True,
# )
# mms_fgm = pyspedas.mms.fgm(
#     trange=trange, probe=probe, data_rate=data_rate, time_clip=True
# )

mms_scm = pyspedas.mms.scm(
    trange=trange, probe=probe, data_rate=data_rate, time_clip=True
)

# # Get raw distribution data
# get_b_str = lambda probe_num: f"mms{probe_num}_fgm_b_gse_{data_rate}_l2"
# time_dist = data_quants[get_b_str(1)].coords["time"].values
# b_data = {}
# for probe_num in range(1, 5):
#     b_string = get_b_str(probe_num)
#     pyspedas.tinterpol(b_string, get_b_str(1), newname=b_string)
#     data = data_quants[b_string].values[:, 3]
#     finiteMask = np.isfinite(data)
#     b_data[f"mms{probe_num}"] = np.interp(
#         time_dist, time_dist[finiteMask], data[finiteMask]
#     )
# # Get time of each step
# # Convert to datetime
# time_dist = np.array([dt.utcfromtimestamp(x) for x in time_dist])

# print(
#     f"""Data loaded. {b_data['mms1'].shape[0]} time steps
# Start:  {dt.strftime(time_dist[0], '%H:%M:%S.%f')}
# End:    {dt.strftime(time_dist[-1], '%H:%M:%S.%f')}"""
# )

# print("Averaging |B| from mms1-4")
# temp = np.empty((b_data["mms1"].shape[0], 4))
# shifts = []
# for i in range(1, 5):
#     print(f"MMS{i}: Assigning data")
#     temp[:, i - 1] = b_data[f"mms{i}"]
#     if i > 1:
#         print(f"MMS{i}: Correlating with MMS1")
#         # See src/shock_normal/timing_analysis.py
#         corr = np.correlate(
#             temp[:, i - 1] - temp[:, i - 1].mean(),
#             temp[:, 0] - temp[:, 0].mean(),
#             mode="full",
#         )
#         shift = np.argmax(corr) - (len(temp[:, i - 1]) - 1)
#         shifts.append(shift)
#         # Shift SC data to same time point
#         print(f"MMS{i}: Shifting {shift} indices")
#         temp[:, i - 1] = shift_with_nan(temp[:, i - 1], shift)

# avg_B = np.mean(temp, axis=1)
# shifts = np.array(shifts)
# # Generate slice to trim nan's created by aligning SC's
# slice_B = slice(
#     max(shifts[shifts > 0]) if len(shifts[shifts > 0]) > 0 else None,
#     -1 + min(shifts[shifts < 0]) if len(shifts[shifts < 0]) > 0 else None,
# )
# avg_B = avg_B[slice_B]
# time_dist = time_dist[slice_B]
# print("Cleaning Memory")
# del temp
# del b_data
# #       l=nu*Deltat+exp(2**x)
# plt.subplot(2, 1, 1)
# plt.plot(time_dist, avg_B)
# plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
# plt.gca().xaxis.set_major_locator(mdates.MinuteLocator(interval=1))

# plt.subplot(2, 1, 2)
# Y = np.fft.fft(avg_B - np.mean(avg_B))
# freq = np.fft.fftfreq(len(avg_B), (time_dist[1] - time_dist[0]).total_seconds())
# # Scale using Taylor Hypothesis

# ν_a = 9.68
# ν_0 = 2 * ν_a
# freq = 2 * np.pi * freq / ν_0

# plt.loglog(freq[freq > 0], abs(Y)[freq > 0])
# plt.xlabel("k[km$^{-1}$]")

# plt.show()
