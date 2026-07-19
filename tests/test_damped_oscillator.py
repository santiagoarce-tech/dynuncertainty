"""Scientific tests for the damped oscillator."""

import unittest

import numpy as np

from dynuncertainty.ensemble import simulate_single_trajectory
from dynuncertainty.models import build_damped_oscillator_model


class DampedOscillatorTests(unittest.TestCase):
    """Verify fundamental damped-oscillator properties."""

    def test_nominal_energy_does_not_increase(self) -> None:
        """Mechanical energy must decrease under positive damping."""

        model = build_damped_oscillator_model()

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

        energy_changes = np.diff(energies)

        numerical_tolerance = (
            1.0e-7 * energies[0]
        )

        self.assertTrue(
            np.all(
                energy_changes <= numerical_tolerance
            )
        )

        self.assertLess(
            energies[-1],
            energies[0],
        )


if __name__ == "__main__":
    unittest.main()