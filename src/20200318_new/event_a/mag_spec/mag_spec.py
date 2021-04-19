import os
import numpy as np
import matplotlib.pyplot as plt
import logging as log
from scipy.ndimage.filters import uniform_filter1d

path = os.path.dirname(os.path.realpath(__file__))
dirpath = "/".join(path.split("/")[:-1])
print(dirpath)

log.basicConfig(
    filename=f"{path}/mag_spec.log",
    level=log.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)

log.info("Loading data")
data = np.load(f"{dirpath}/data/fsm/data.npy")
log.info("Loading time")
time = np.load(f"{dirpath}/data/fsm/time.npy")
td = time[1] - time[0]

Y = {}
log.info("Comutping FFT over each coord")
for i in range(3):
    log.info(f"index {i}")
    B = data[:, i] * 1e-9
    log.info("Scaling mean")
    B -= B.mean()

    log.info("Applying Hanning window")
    Hann = np.hanning(len(B)) * B
    log.info("Calculating FFT")
    Yi = np.fft.fft(Hann)
    log.info("Calculating Frequencies")
    freq = np.fft.fftfreq(len(B), td)
    log.info("Obtaining power spectrum")
    Y[["x", "y", "z"][i]] = (np.power(np.abs(Yi), 2) * 1e9 * td)[freq > 0]
log.info("Summing components")
y = np.sum([Y[i] for i in ["x", "y", "z"]], axis=0)
freq = freq[freq > 0]

log.info("Saving...")
np.save(f"{path}/total_y.npy", y)
np.save(f"{path}/total_freq.npy", freq)

# log.info("Plotting...")
# plt.loglog(freq, y)
# plt.xlabel("Frequency [Hz]")
# plt.ylabel("Magnetic Spectrum [$nT^2Hz^{-1}$]")

# plt.savefig(f"{path}/mag_spec/mag_spec.png")