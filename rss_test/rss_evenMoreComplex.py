from rss_simple import timer, speedTest
import numpy as np


@timer
def rss3(a: np.ndarray) -> float:
    return np.linalg.norm(a)


speedTest(rss3, 1000, 10000)

# Output
# Computing the root sum square of 10000 items, over 1000 tests
# Average :   0.570 ms
# Sigma   :   0.032ms