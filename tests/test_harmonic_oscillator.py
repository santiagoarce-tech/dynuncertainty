"""Scientific tests for the harmonic oscillator."""

import unittest

import numpy as np

from dynuncertainty.ensemble import simulate_single_trajectory
from dynuncertainty.models import build_harmonic_oscillator_model


class HarmonicOscillatorTests(unittest.TestCase):
    """Verify fundamental harmonic-oscillator properties."""

    def test_nominal_energy_is_conserved(self) -> None:
        """The nominal mechanical energy must remain constant."""

        model = build_harmonic_oscillator_model()

        parameters = {
            parameter.name: parameter.nominal
            for parameter in model.parameters
        }

        initial_state = np.array(
            [
                condition.nominal
                for condition in model.initial_conditions
            ],
            dtype=float,
        )

        times = np.linspace(
            model.default_t_start,
            model.default_t_end,
            model.default_time_points,
        )

        trajectory = simulate_single_trajectory(
            model=model.derivative,
            x0=initial_state,
            params=parameters,
            t_span=(times[0], times[-1]),
            t_eval=times,
        )

        positions = trajectory[:, 0]
        velocities = trajectory[:, 1]

        mass = parameters["m"]
        stiffness = parameters["k"]

        energies = (
            0.5 * mass * velocities**2
            + 0.5 * stiffness * positions**2
        )

        relative_energy_drift = np.max(
            np.abs(energies - energies[0])
        ) / energies[0]

        self.assertLess(
            relative_energy_drift,
            1.0e-6,
        )


if __name__ == "__main__":
    unittest.main()