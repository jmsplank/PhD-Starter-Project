from rss_simple import timer, speedTest
import numpy as np


@timer
def rss2(a: np.ndarray) -> float:
    """Evaluates sqrt(a[0]**2+a[1]**2 + ...)"""
    return np.sqrt(np.sum(np.power(a, 2)))


speedTest(rss2, 1000, 10000)

# Output:
# Computing the root sum square of 10000 items, over 1000 tests
# Average :   0.779 ms
# Sigma   :   0.144ms