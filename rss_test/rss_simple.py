from math import sqrt
import numpy as np

import functools  # For timing decorator
import time

from typing import Callable, Tuple, List  # Py3.8 type hints


def timer(func: Callable) -> Tuple[Callable, float]:
    """Time the execution of a function."""

    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        tic = time.perf_counter()
        value = func(*args, **kwargs)
        toc = time.perf_counter()
        elapsed = toc - tic
        return value, elapsed

    return wrapper_timer


@timer
def rss(a: List) -> float:
    """Evaluates sqrt(a[0]**2+a[1]**2 + ...)"""
    squares = []
    for i in range(len(a)):
        squares.append(a[i] ** 2)
    sum_a = sum(squares)
    return sqrt(sum_a)


def speedTest(f: Callable, num_tests: int, len_list: int) -> None:
    tests = []
    for _ in range(num_tests):
        a = list(np.random.rand(len_list))
        tests.append(f(a)[1])

    print(
        f"""Computing the root sum square of {len_list} items, over {num_tests} tests
Average :   {np.mean(tests)*1000:0.3f} ms
Sigma   :   {np.std(tests)*1000:0.3f}ms"""
    )


if __name__ == "__main__":
    speedTest(1000, 10000)

# Output:
# Computing the root sum square of 10000 items, over 1000 tests
# Average :   4.207 ms
# Sigma   :   0.198ms