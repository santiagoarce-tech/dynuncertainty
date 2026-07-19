"""Tests for dynuncertainty ensemble execution."""

import unittest

import numpy as np

from dynuncertainty.ensemble import (
    simulate_ensemble,
    simulate_ensemble_sequential,
)
from dynuncertainty.models import build_harmonic_oscillator_model


class EnsembleTests(unittest.TestCase):
    """Verify the production ensemble engine."""

    def test_chunked_engine_matches_sequential_reference(
        self,
    ) -> None:
        """Chunked trajectories must match the reference engine."""

        model = build_harmonic_oscillator_model()

        n_samples = 25

        initial_conditions = np.column_stack(
            (
                np.linspace(0.95, 1.05, n_samples),
                np.linspace(-0.05, 0.05, n_samples),
            )
        )

        masses = np.linspace(
            0.95,
            1.05,
            n_samples,
        )

        stiffnesses = np.linspace(
            1.05,
            0.95,
            n_samples,
        )

        parameter_samples = [
            {
                "m": float(mass),
                "k": float(stiffness),
            }
            for mass, stiffness in zip(
                masses,
                stiffnesses,
            )
        ]

        times = np.linspace(
            0.0,
            5.0,
            101,
        )

        chunked_trajectories = simulate_ensemble(
            model=model.derivative,
            initial_conditions=initial_conditions,
            parameter_samples=parameter_samples,
            t_span=(times[0], times[-1]),
            t_eval=times,
        )

        sequential_trajectories = (
            simulate_ensemble_sequential(
                model=model.derivative,
                initial_conditions=initial_conditions,
                parameter_samples=parameter_samples,
                t_span=(times[0], times[-1]),
                t_eval=times,
            )
        )

        self.assertEqual(
            chunked_trajectories.shape,
            (n_samples, len(times), 2),
        )

        self.assertTrue(
            np.all(np.isfinite(chunked_trajectories))
        )

        np.testing.assert_allclose(
            chunked_trajectories,
            sequential_trajectories,
            rtol=1.0e-5,
            atol=1.0e-6,
        )


if __name__ == "__main__":
    unittest.main()