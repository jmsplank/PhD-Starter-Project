import numpy as np
import matplotlib.pyplot as plt
from phdhelper.helpers import override_mpl
from phdhelper.helpers.os_shortcuts import get_path
from phdhelper.helpers import COLOURS
from tqdm import tqdm
from scipy.stats import kurtosis
import logging
from matplotlib.colors import LogNorm

override_mpl.override()
override_mpl.cmaps("custom_diverging")

path = get_path(__file__)
path2 = get_path(__file__, "..")

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.FileHandler(f"{path}/log.log", "w")
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

big_data = np.load(f"{path2}/data/fsm/data.npy")
big_time = np.load(f"{path2}/data/fsm/time.npy")

p_max = 4
Ni = 10 ** (4 + 1)  # (Dudock de Wit et al. 2013)
logger.info(f"Length of bins: {Ni}")
N = len(big_time) // Ni
logger.info(f"Number of bins: {N}")
logger.info(f"Length of bins (s): {big_time[Ni]-big_time[0]:0.4f}s")

max_index = len(big_time) - Ni
bin_starts = np.linspace(0, max_index, N, dtype=int)

num_samples = 100  # Number of lags
lags_array = np.logspace(0, np.log10(Ni), num_samples, dtype=int)

kurt = np.empty(N)  # Container for scale independednt kurtosis @ window
bin_times = np.empty(N)  # Container for time @ each window
big_K = np.empty((num_samples, N))  # 2d container for lags @ each window
max_K = np.empty(N)  # container for max kurtosis @ each window

for i, bin in enumerate(tqdm(bin_starts)):  # Loop over windows
    data = big_data[bin : bin + Ni, :].mean(axis=1)  # B field
    time = big_time[bin : bin + Ni]  # time

    bin_times[i] = time[len(time) // 2]  # Get avg (center) time for window

    data_norm = data - data.mean()  # subtract mean
    kurt[i] = kurtosis(data_norm, fisher=False)  # calculate indep. kurtosis

    increment = np.empty((len(data_norm), num_samples))  # 2d (all B, all lags)
    for lag_index, lag in enumerate(lags_array):
        # b(t) - b(t + lag) by rolling array by -lag indices
        increment[:, lag_index] = data_norm - np.roll(data_norm, -lag)

    fourth = np.power(increment, 4).mean(axis=0)  # avg over B
    second = np.power(increment, 2).mean(axis=0)  # avg over B

    K = fourth / np.power(second, 2)  # ratio of 4th & second moment => kurtosis
    big_K[:, i] = K  # assign kurtosis(lag) to container
    max_K[i] = max(K)  # assign maximum kurtosis to container

    if i == 0:  # debug
        logger.info(f"data_sampled: {data_norm}")
        logger.info(f"sd_kurt:\n{increment}")
        logger.info(
            f"sd_kurt %NAN: {np.count_nonzero(np.isnan(increment))/increment.size*100}%"
        )
        logger.info(f"max_k: {max_K[i]:0.3f}")
        fig, ax = plt.subplots(3, 1, sharex=True)
        for i in range(3):
            ax[i].plot(lags_array, [fourth, second, K][i])
            ax[i].set_ylabel(["fourth", "second", "K"][i])
            ax[i].set_xscale("log")
            ax[i].set_yscale("log")

        plt.savefig(f"{path}/log_i_0.png")
        plt.close()


fig, ax = plt.subplots(3, 1, sharex=True, figsize=(8, 6))

ax[0].plot(big_time - big_time[0], np.linalg.norm(big_data, axis=1), color="k")

par1 = ax[1].twinx()  # parasite axis (plot 2 scales on ax[1])

# scale independent
(p1,) = ax[1].plot(bin_times - bin_times[0], kurt, color=COLOURS.red)
# scale dependent
(p2,) = par1.plot(bin_times - bin_times[0], max_K, color=COLOURS.green)

ax[1].axhline(3, color="k")
ax[1].set_ylabel("Scale independent kurtosis")
par1.set_ylabel("Max scale dep kurtosis")

ax[1].yaxis.label.set_color(p1.get_color())
par1.yaxis.label.set_color(p2.get_color())


ax[2].pcolor(
    bin_times - bin_times[0],
    lags_array,
    big_K,
    norm=LogNorm(),
    shading="nearest",
)
ax[2].set_yscale("log")

ax[0].set_ylabel("$|B|$")
ax[2].set_ylabel("Lag (index)")
ax[2].set_xlabel("Time (s)")

logger.info(f"max_K: mean: {max_K.mean()} std: {max_K.std()}")
logger.info(f"big_K: mean: {big_K.mean()} std: {big_K.std()}")
np.save(f"{path}/max_K.npy", max_K)
np.save(f"{path}/big_K.npy", big_K)
np.save(f"{path}/bin_times.npy", bin_times)
np.save(f"{path}/kurt.npy", kurt)

plt.tight_layout()
plt.show()