import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import periodogram
from datetime import datetime as dt

saved_data = np.load("src/magSpec/npy/index_function.npy", allow_pickle=True).item()

x = [dt.utcfromtimestamp(t) for t in saved_data["x"]]
print(x[1], x[0])
xs = (x[1] - x[0]).total_seconds()
print(xs)
fs = 1.0 / xs
print(fs)
y = saved_data["out"]

for i in range(3):
    f, Pxx = periodogram(
        y[:, i],
        fs,
        window="hanning",
        scaling="spectrum",
    )
    plt.subplot(3, 1, i + 1)
    plt.plot(f, Pxx)
    plt.yscale("log")
    plt.xlabel("Frequency (samples/second)")
    plt.ylabel("Periodogram spectrum amplitude")

plt.tight_layout()
plt.show()
