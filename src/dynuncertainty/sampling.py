import numpy as np


def sample_initial_conditions(intervals, n_samples, rng=None):
    """
    Sample initial conditions from intervals.

    Example:
        intervals = [
            (0.95, 1.05),   # x0
            (-0.02, 0.02),  # v0
        ]
    """
    rng = np.random.default_rng(rng)

    samples = np.zeros((n_samples, len(intervals)))

    for i, (low, high) in enumerate(intervals):
        samples[:, i] = rng.uniform(low, high, size=n_samples)

    return samples


def sample_parameter_intervals(parameter_intervals, n_samples, rng=None):
    """
    Sample parameters from intervals.

    Example:
        parameter_intervals = {
            "m": (0.95, 1.05),
            "k": (0.95, 1.05),
        }
    """
    rng = np.random.default_rng(rng)

    samples = []

    for _ in range(n_samples):
        params = {}

        for name, (low, high) in parameter_intervals.items():
            params[name] = rng.uniform(low, high)

        samples.append(params)

    return samples