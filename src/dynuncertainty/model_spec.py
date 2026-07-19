"""
Common data structures used to describe dynamical models.

This module defines the contract between the model catalogue,
the interactive interface, and the numerical simulation engine.
"""

from dataclasses import dataclass, field
from typing import Callable, Literal

import numpy as np
from numpy.typing import NDArray


State = NDArray[np.float64]
ParameterValues = dict[str, float]
DerivativeFunction = Callable[[float, State, ParameterValues], State]
PhaseReferenceFrequencyFunction = Callable[
    [ParameterValues],
    float,
]

PhaseBandStrategy = Literal[
    "radial",
    "trajectory_tube",
    "angular_energy",
]

@dataclass(frozen=True)
class ParameterSpec:
    """
    Description of one uncertain model parameter.

    Parameters
    ----------
    name
        Internal parameter name used by the differential equation.
    label
        Human-readable label displayed by the interface.
    nominal
        Reference or nominal value.
    lower
        Lower uncertainty bound.
    upper
        Upper uncertainty bound.
    unit
        Physical unit displayed by the interface.
    """

    name: str
    label: str
    nominal: float
    lower: float
    upper: float
    unit: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Parameter name cannot be empty.")

        if self.lower > self.nominal:
            raise ValueError(
                f"Lower bound of '{self.name}' cannot exceed its nominal value."
            )

        if self.nominal > self.upper:
            raise ValueError(
                f"Nominal value of '{self.name}' cannot exceed its upper bound."
            )


@dataclass(frozen=True)
class InitialConditionSpec:
    """
    Description of one uncertain initial-state component.

    Parameters
    ----------
    name
        Internal name of the initial condition.
    label
        Human-readable label displayed by the interface.
    nominal
        Reference initial value.
    lower
        Lower uncertainty bound.
    upper
        Upper uncertainty bound.
    unit
        Physical unit displayed by the interface.
    """

    name: str
    label: str
    nominal: float
    lower: float
    upper: float
    unit: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Initial-condition name cannot be empty.")

        if self.lower > self.nominal:
            raise ValueError(
                f"Lower bound of '{self.name}' cannot exceed "
                "its nominal value."
            )

        if self.nominal > self.upper:
            raise ValueError(
                f"Nominal value of '{self.name}' cannot exceed "
                "its upper bound."
            )

@dataclass(frozen=True)
class ModelSpec:
    """
    Complete description of one model in the catalogue.

    The interface reads this object to determine which controls
    must be created. The simulation engine uses ``derivative`` to
    evolve the state.
    """

    key: str
    name: str
    derivative: DerivativeFunction
    parameters: tuple[ParameterSpec, ...]
    initial_conditions: tuple[InitialConditionSpec, ...]
    state_labels: tuple[str, ...]
    phase_reference_frequency: ( PhaseReferenceFrequencyFunction | None) = None
    phase_band_strategy: PhaseBandStrategy = "radial"
    default_t_start: float = 0.0
    default_t_end: float = 30.0
    default_time_points: int = 1000
    default_n_samples: int = 200
    default_seed: int = 42
    description: str = ""
    metadata: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.key:
            raise ValueError("Model key cannot be empty.")

        if not self.name:
            raise ValueError("Model name cannot be empty.")

        if self.default_t_end <= self.default_t_start:
            raise ValueError(
                "The final simulation time must exceed the initial time."
            )

        if self.default_time_points < 2:
            raise ValueError(
                "A simulation requires at least two time points."
            )
        
        if self.default_n_samples < 1:
            raise ValueError(
                "The number of ensemble samples must be at least 1."
            )

        if self.default_seed < 0:
            raise ValueError(
                "The random seed cannot be negative."
            )

        if len(self.initial_conditions) != len(self.state_labels):
            raise ValueError(
                "Each initial-state component must have a state label."
            )

        parameter_names = [parameter.name for parameter in self.parameters]

        if len(parameter_names) != len(set(parameter_names)):
            raise ValueError("Parameter names must be unique.")

    @property
    def nominal_parameters(self) -> ParameterValues:
        """Return all nominal parameter values as a dictionary."""

        return {
            parameter.name: parameter.nominal
            for parameter in self.parameters
        }

    @property
    def parameter_intervals(self) -> dict[str, tuple[float, float]]:
        """Return the lower and upper bounds of every parameter."""

        return {
            parameter.name: (parameter.lower, parameter.upper)
            for parameter in self.parameters
        }

    @property
    def default_initial_state(self) -> State:
        """Return the default initial state as a NumPy array."""

        return np.array(
            [
                condition.nominal
                for condition in self.initial_conditions
            ],
            dtype=float,
        )