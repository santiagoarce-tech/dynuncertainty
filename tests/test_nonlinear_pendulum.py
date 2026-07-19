"""Scientific tests for the nonlinear pendulum."""

import unittest

import numpy as np

from dynuncertainty.ensemble import simulate_single_trajectory
from dynuncertainty.models import build_nonlinear_pendulum_model


class NonlinearPendulumTests(unittest.TestCase):
    """Verify fundamental nonlinear-pendulum properties."""

    def test_nominal_energy_is_conserved(self) -> None:
        """The nominal pendulum energy must remain constant."""

        model = build_nonlinear_pendulum_model()

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

        angles = trajectory[:, 0]
        angular_velocities = trajectory[:, 1]

        gravity = parameters["g"]
        length = parameters["L"]

        energies = (
            0.5 * length**2 * angular_velocities**2
            + gravity
            * length
            * (1.0 - np.cos(angles))
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