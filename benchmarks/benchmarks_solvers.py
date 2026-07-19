"""Reproducible solver benchmark for dynuncertainty."""

from time import perf_counter

import numpy as np

from dynuncertainty.ensemble import (
    simulate_ensemble_chunked,
    simulate_ensemble_collective,
    simulate_ensemble_sequential,
)

from dynuncertainty.models import (
    build_nonlinear_pendulum_model,
)



METHOD = "DOP853"

ENGINES = {
    "sequential": simulate_ensemble_sequential,
    "collective": simulate_ensemble_collective,
    "chunked": simulate_ensemble_chunked,
}

COMPARISON_ENGINES = (
    "collective",
    "chunked",
)

PENDULUM_ENERGY_RANGES = {
    "libration": (0.30, 0.80),
    "near_separatrix": (0.9995, 1.0005),
    "positive_rotation": (1.05, 1.25),
    "negative_rotation": (1.05, 1.25),
}


def create_pendulum_regime_samples(
    model,
    n_samples,
    seed,
    regime,
):
    """Create pendulum samples inside one energy regime."""

    if regime not in PENDULUM_ENERGY_RANGES:
        raise ValueError(
            f"Unknown pendulum regime: {regime!r}."
        )

    rng = np.random.default_rng(seed)

    gravity_lower, gravity_upper = (
        model.parameter_intervals["g"]
    )

    length_lower, length_upper = (
        model.parameter_intervals["L"]
    )

    initial_intervals = {
        condition.name: (
            condition.lower,
            condition.upper,
        )
        for condition in model.initial_conditions
    }

    angle_lower, angle_upper = (
        initial_intervals["theta0"]
    )

    gravity_values = rng.uniform(
        gravity_lower,
        gravity_upper,
        size=n_samples,
    )

    length_values = rng.uniform(
        length_lower,
        length_upper,
        size=n_samples,
    )

    initial_angles = rng.uniform(
        angle_lower,
        angle_upper,
        size=n_samples,
    )

    ratio_lower, ratio_upper = (
        PENDULUM_ENERGY_RANGES[regime]
    )

    target_energy_ratios = rng.uniform(
        ratio_lower,
        ratio_upper,
        size=n_samples,
    )

    potential_energy_ratios = (
        0.5 * (1.0 - np.cos(initial_angles))
    )

    angular_speed_magnitudes = np.sqrt(
        (4.0 * gravity_values / length_values)
        * (
            target_energy_ratios
            - potential_energy_ratios
        )
    )

    if regime == "positive_rotation":
        directions = np.ones(n_samples)
    elif regime == "negative_rotation":
        directions = -np.ones(n_samples)
    else:
        directions = rng.choice(
            (-1.0, 1.0),
            size=n_samples,
        )

    initial_angular_velocities = (
        directions * angular_speed_magnitudes
    )

    initial_samples = np.column_stack(
        (
            initial_angles,
            initial_angular_velocities,
        )
    )

    parameter_samples = [
        {
            "g": float(gravity),
            "L": float(length),
        }
        for gravity, length in zip(
            gravity_values,
            length_values,
        )
    ]

    return (
        initial_samples,
        parameter_samples,
        target_energy_ratios,
    )

def time_engine(
    engine,
    model,
    initial_samples,
    parameter_samples,
    t_eval,
):
    """Time one execution engine without sample generation."""

    started_at = perf_counter()

    trajectories = engine(
        model=model.derivative,
        initial_conditions=initial_samples,
        parameter_samples=parameter_samples,
        t_span=(float(t_eval[0]), float(t_eval[-1])),
        t_eval=t_eval,
        method=METHOD,
    )

    elapsed = perf_counter() - started_at

    return elapsed, trajectories

def compute_energy_ratio_drift(
    trajectories,
    parameter_samples,
):
    """Measure numerical drift in pendulum energy ratio."""

    angles = trajectories[:, :, 0]
    angular_velocities = trajectories[:, :, 1]

    gravity_values = np.array(
        [
            parameters["g"]
            for parameters in parameter_samples
        ],
        dtype=float,
    )[:, None]

    length_values = np.array(
        [
            parameters["L"]
            for parameters in parameter_samples
        ],
        dtype=float,
    )[:, None]

    specific_energies = (
        0.5
        * length_values**2
        * angular_velocities**2
        + gravity_values
        * length_values
        * (1.0 - np.cos(angles))
    )

    separatrix_energies = (
        2.0
        * gravity_values
        * length_values
    )

    energy_ratios = (
        specific_energies / separatrix_energies
    )

    energy_ratio_drift = (
        energy_ratios - energy_ratios[:, [0]]
    )

    maximum_absolute_drift = float(
        np.max(np.abs(energy_ratio_drift))
    )

    root_mean_square_drift = float(
        np.sqrt(np.mean(energy_ratio_drift**2))
    )

    return maximum_absolute_drift, root_mean_square_drift

def main():
    """Compare all integration engines across pendulum regimes."""

    model = build_nonlinear_pendulum_model()

    n_samples = 200
    seed = 42

    t_eval = np.linspace(
        model.default_t_start,
        model.default_t_end,
        model.default_time_points,
    )

    print("\ndynuncertainty solver benchmark")
    print(f"Model: {model.name}")
    print(f"Samples per regime: {n_samples}")
    print(f"Seed: {seed}")
    print(f"Time points: {len(t_eval)}")

    for regime in PENDULUM_ENERGY_RANGES:
        (
            initial_samples,
            parameter_samples,
            target_energy_ratios,
        ) = create_pendulum_regime_samples(
            model=model,
            n_samples=n_samples,
            seed=seed,
            regime=regime,
        )

        print("\n" + "=" * 60)
        print(f"Regime: {regime}")
        print(
            "Target ρ range: "
            f"[{np.min(target_energy_ratios):.6f}, "
            f"{np.max(target_energy_ratios):.6f}]"
        )

        trajectory_sets: dict[str, np.ndarray] = {}

        for engine_name, engine in ENGINES.items():
            elapsed, trajectories = time_engine(
                engine=engine,
                model=model,
                initial_samples=initial_samples,
                parameter_samples=parameter_samples,
                t_eval=t_eval,
            )

            trajectory_sets[engine_name] = trajectories

            (
                maximum_energy_drift,
                root_mean_square_energy_drift,
            ) = compute_energy_ratio_drift(
                trajectories=trajectories,
                parameter_samples=parameter_samples,
            )

            print(
                f"{engine_name:>10}: "
                f"{elapsed:.3f} s, "
                f"shape={trajectories.shape}"
            )

            print(
                "         "
                f"max |Δρ|={maximum_energy_drift:.3e}, "
                f"RMS Δρ="
                f"{root_mean_square_energy_drift:.3e}"
            )

        for candidate_name in COMPARISON_ENGINES:
            trajectory_difference = (
                trajectory_sets["sequential"]
                - trajectory_sets[candidate_name]
            )

            maximum_absolute_difference = float(
                np.max(np.abs(trajectory_difference))
            )

            root_mean_square_difference = float(
                np.sqrt(
                    np.mean(trajectory_difference**2)
                )
            )

            per_trajectory_maximum_differences = np.max(
                np.abs(trajectory_difference),
                axis=(1, 2),
            )

            median_trajectory_difference = float(
                np.percentile(
                    per_trajectory_maximum_differences,
                    50,
                )
            )

            percentile_95_trajectory_difference = float(
                np.percentile(
                    per_trajectory_maximum_differences,
                    95,
                )
            )

            percentile_99_trajectory_difference = float(
                np.percentile(
                    per_trajectory_maximum_differences,
                    99,
                )
            )

            worst_trajectory_index = int(
                np.argmax(
                    per_trajectory_maximum_differences
                )
            )

            worst_trajectory_energy_ratio = float(
                target_energy_ratios[
                    worst_trajectory_index
                ]
            )

            print(
                "\nNumerical comparison: "
                f"sequential vs {candidate_name}"
            )
            print(
                "  Maximum absolute difference: "
                f"{maximum_absolute_difference:.3e}"
            )
            print(
                "  Root-mean-square difference: "
                f"{root_mean_square_difference:.3e}"
            )
            print(
                "  Median trajectory maximum: "
                f"{median_trajectory_difference:.3e}"
            )
            print(
                "  95th percentile: "
                f"{percentile_95_trajectory_difference:.3e}"
            )
            print(
                "  99th percentile: "
                f"{percentile_99_trajectory_difference:.3e}"
            )
            print(
                "  Worst trajectory index: "
                f"{worst_trajectory_index}"
            )
            print(
                "  Worst trajectory target ρ: "
                f"{worst_trajectory_energy_ratio:.9f}"
            )


if __name__ == "__main__":
    main()