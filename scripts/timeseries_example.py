
import scipy.optimize as opt
from typing import List
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from time import time
import random

mpl.style.use("seaborn-dark")


def jumps_to_intervals(jumps, *, x_max=lambda j_max: j_max + 1, samples_per_segment=2):
    j = np.array([0, *jumps[:], x_max(jumps[-1])])
    intervals = np.vstack(np.linspace(
        j[:-1], j[1:], num=samples_per_segment)).T  # start, end pairs
    return intervals


def constrained_partition(total, number_of_parts):
    """Generates a random (not necessarily uniformly distributed) integer partition."""
    remaining_parts_count = number_of_parts
    parts = []
    remaining_total = total
    for _ in range(number_of_parts-1):
        # new_part = random.randint(1, remaining_total - remaining_parts_count + 1)
        t = 0.8
        new_part = int(np.round(random.triangular(1, remaining_total - remaining_parts_count +
                       1, ((1-t) * (remaining_total - remaining_parts_count + 1) + t))))
        parts.append(new_part)
        remaining_total -= new_part
        remaining_parts_count -= 1
    parts.append(remaining_total)
    random.shuffle(parts)
    return parts


def sample_signal(n_jumps, length, noise_factor=0.005, jump_seed=None, data_seed=None, noise_seed=None):
    """Synthetic piecewise constant data"""
    if jump_seed is not None:
        random.seed(jump_seed)
    lens = constrained_partition(length, n_jumps + 1)
    if data_seed is not None:
        np.random.seed(data_seed)
    data = np.hstack([np.random.uniform(0, 1) * np.ones(l) for l in lens])
    if noise_seed is not None:
        np.random.seed(noise_seed)
    data += noise_factor * np.random.randn(length)
    return data


def sample_signal_poly(n_jumps, max_total_dofs, max_segment_dofs, length, noise_factor=0.005, jump_seed=None, data_seed=None, noise_seed=None):
    """Synthetic piecewise constant data"""
    if jump_seed is not None:
        random.seed(jump_seed)
    lens = constrained_partition(length, n_jumps + 1)
    dofs = constrained_partition(max_total_dofs, n_jumps + 1)
    if data_seed is not None:
        np.random.seed(data_seed)
    polys = (np.polynomial.Polynomial(-5*np.random.randn(np.random.randint(1, min(max_segment_dofs, d) + 1)))
             for d in dofs)
    c = np.cumsum(lens)
    bounds = np.vstack([np.hstack([0, c[:-1]]), c]).T
    xs = np.linspace(-1, 1, num=length)
    data = np.hstack([p(xs[b[0]:b[1]]) for p, b in zip(polys, bounds)])

    a = 2 / (np.amax(data) - np.amin(data))
    b = (1 - a * np.amax(data))
    data = a * data + b
    if noise_seed is not None:
        np.random.seed(noise_seed)
    data += noise_factor * np.random.randn(length)
    return xs, data

xs, ys = sample_signal_poly(3, 30, 6, 1000, noise_factor=0, jump_seed=1, data_seed=1, noise_seed=0)

ax = plt.subplot()
ax.scatter((xs[::10]+1)/2, ys[::10], label="Signal sample")
ax.plot((xs+1)/2, ys, label="Signal")
ax.legend()
ax.grid()
ax.set_xlabel("Time")
plt.show()