import numpy as np
from scipy.integrate import solve_ivp


def simulate_single_trajectory(model, x0, params, t_span, t_eval, method="DOP853"):
    """
    Simulate one trajectory of an ODE system.
    """
    solution = solve_ivp(
        fun=lambda t, x: model(t, x, params),
        t_span=t_span,
        y0=x0,
        t_eval=t_eval,
        method=method,
        rtol=1e-8,
        atol=1e-10,
    )

    if not solution.success:
        raise RuntimeError(f"ODE solver failed: {solution.message}")

    return solution.y.T


def simulate_ensemble_sequential(
    model,
    initial_conditions,
    parameter_samples,
    t_span,
    t_eval,
    method="DOP853",
):
    """
    Simulate many trajectories.

    Returns:
        trajectories with shape:
        (n_samples, n_times, state_dimension)
    """
    if len(initial_conditions) != len(parameter_samples):
        raise ValueError("initial_conditions and parameter_samples must have the same length.")

    trajectories = []

    for x0, params in zip(initial_conditions, parameter_samples):
        trajectory = simulate_single_trajectory(
            model=model,
            x0=x0,
            params=params,
            t_span=t_span,
            t_eval=t_eval,
            method=method,
        )

        trajectories.append(trajectory)

    return np.array(trajectories)

def simulate_ensemble_collective(
    model,
    initial_conditions,
    parameter_samples,
    t_span,
    t_eval,
    method="DOP853",
):
    """
    Simulate an ensemble as one collective ODE system.

    Every trajectory remains dynamically independent, but all
    trajectories share one adaptive integration process.
    """

    initial_array = np.asarray(
        initial_conditions,
        dtype=float,
    )

    if initial_array.ndim != 2:
        raise ValueError(
            "Initial conditions must have shape "
            "(n_samples, state_dimension)."
        )

    n_samples, state_dimension = initial_array.shape

    if n_samples < 1:
        raise ValueError(
            "A collective ensemble requires at least one sample."
        )

    if len(parameter_samples) != n_samples:
        raise ValueError(
            "Initial conditions and parameter samples must "
            "have the same length."
        )

    parameter_names = tuple(parameter_samples[0])

    expected_parameter_names = set(parameter_names)

    for parameters in parameter_samples:
        if set(parameters) != expected_parameter_names:
            raise ValueError(
                "Every parameter sample must contain the "
                "same parameter names."
            )

    collective_parameters = {
        name: np.array(
            [
                parameters[name]
                for parameters in parameter_samples
            ],
            dtype=float,
        )
        for name in parameter_names
    }

    collective_initial_state = (
        initial_array.T.reshape(-1)
    )

    def collective_derivative(t, flat_state):
        state_matrix = flat_state.reshape(
            state_dimension,
            n_samples,
        )

        derivative_matrix = np.asarray(
            model(
                t,
                state_matrix,
                collective_parameters,
            ),
            dtype=float,
        )

        if derivative_matrix.shape != state_matrix.shape:
            raise ValueError(
                "The model derivative does not support "
                "collective array evaluation."
            )

        return derivative_matrix.reshape(-1)

    solution = solve_ivp(
        fun=collective_derivative,
        t_span=t_span,
        y0=collective_initial_state,
        t_eval=t_eval,
        method=method,
        rtol=1e-8,
        atol=1e-10,
    )

    if not solution.success:
        raise RuntimeError(
            f"Collective ODE solver failed: "
            f"{solution.message}"
        )

    trajectories = solution.y.reshape(
        state_dimension,
        n_samples,
        len(solution.t),
    ).transpose(1, 2, 0)

    return trajectories

def simulate_ensemble_chunked(
    model,
    initial_conditions,
    parameter_samples,
    t_span,
    t_eval,
    method="DOP853",
    chunk_size=20,
):
    """
    Simulate an ensemble in collective chunks.

    Each chunk has its own adaptive integration process. This
    preserves much of the collective speed while limiting the
    number of trajectories controlled by one global error norm.
    """

    initial_array = np.asarray(
        initial_conditions,
        dtype=float,
    )

    if initial_array.ndim != 2:
        raise ValueError(
            "Initial conditions must have shape "
            "(n_samples, state_dimension)."
        )

    n_samples = initial_array.shape[0]

    if n_samples < 1:
        raise ValueError(
            "A chunked ensemble requires at least one sample."
        )

    if len(parameter_samples) != n_samples:
        raise ValueError(
            "Initial conditions and parameter samples must "
            "have the same length."
        )

    if not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError(
            "Chunk size must be a positive integer."
        )

    parameter_list = list(parameter_samples)

    trajectory_chunks = []

    for chunk_start in range(
        0,
        n_samples,
        chunk_size,
    ):
        chunk_end = min(
            chunk_start + chunk_size,
            n_samples,
        )

        chunk_trajectories = (
            simulate_ensemble_collective(
                model=model,
                initial_conditions=initial_array[
                    chunk_start:chunk_end
                ],
                parameter_samples=parameter_list[
                    chunk_start:chunk_end
                ],
                t_span=t_span,
                t_eval=t_eval,
                method=method,
            )
        )

        trajectory_chunks.append(
            chunk_trajectories
        )

    return np.concatenate(
        trajectory_chunks,
        axis=0,
    )

def simulate_ensemble(
    model,
    initial_conditions,
    parameter_samples,
    t_span,
    t_eval,
    method="DOP853",
):
    """
    Simulate an ensemble in collective chunks of 20 samples.

    The sequential and fully collective implementations remain
    available as scientific reference and benchmarking engines.
    """

    return simulate_ensemble_chunked(
        model=model,
        initial_conditions=initial_conditions,
        parameter_samples=parameter_samples,
        t_span=t_span,
        t_eval=t_eval,
        method=method,
        chunk_size=20,
    )