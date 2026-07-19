import numpy as np

from dynuncertainty.model_spec import (
    InitialConditionSpec,
    ModelSpec,
    ParameterSpec,
)

def spring_mass_reference_frequency(
    parameters: dict[str, float],
) -> float:
    """Return the reference frequency for a spring-mass system."""

    mass = parameters["m"]
    stiffness = parameters["k"]

    return float(np.sqrt(stiffness / mass))


def pendulum_reference_frequency(
    parameters: dict[str, float],
) -> float:
    """Return the small-angle reference frequency of a pendulum."""

    gravity = parameters["g"]
    length = parameters["L"]

    return float(np.sqrt(gravity / length))


def harmonic_oscillator(t, state, params):
    """
    Harmonic oscillator with uncertain mass and spring constant.

    State:
        state = [x, v]

    Parameters:
        params["m"] : mass
        params["k"] : spring constant
    """
    x, v = state

    m = params["m"]
    k = params["k"]

    dxdt = v
    dvdt = -(k / m) * x

    return np.array([dxdt, dvdt])

def damped_oscillator(t, state, params):
    """
    Damped harmonic oscillator.

    State:
        state = [x, v]

    Parameters:
        params["m"] : mass
        params["k"] : spring constant
        params["c"] : damping coefficient
    """

    x, v = state

    m = params["m"]
    k = params["k"]
    c = params["c"]

    dxdt = v
    dvdt = -(k / m) * x - (c / m) * v

    return np.array([dxdt, dvdt])

def nonlinear_pendulum(t, state, params):
    """
    Nonlinear simple pendulum.

    State:
        state = [theta, omega]

    Parameters:
        params["g"] : gravitational acceleration
        params["L"] : pendulum length
    """

    theta, omega = state

    gravity = params["g"]
    length = params["L"]

    dtheta_dt = omega
    domega_dt = -(gravity / length) * np.sin(theta)

    return np.array([dtheta_dt, domega_dt])

def build_damped_oscillator_model() -> ModelSpec:
    """Build the damped-oscillator catalogue entry."""

    return ModelSpec(
        key="damped_oscillator",
        name="Damped oscillator",
        derivative=damped_oscillator,
        parameters=(
            ParameterSpec(
                name="m",
                label="Mass",
                nominal=1.0,
                lower=0.95,
                upper=1.05,
                unit="kg",
            ),
            ParameterSpec(
                name="k",
                label="Spring constant",
                nominal=1.0,
                lower=0.95,
                upper=1.05,
                unit="N/m",
            ),
            ParameterSpec(
                name="c",
                label="Damping coefficient",
                nominal=0.15,
                lower=0.10,
                upper=0.20,
                unit="N s/m",
            ),
        ),
        initial_conditions=(
            InitialConditionSpec(
                name="x0",
                label="Initial position",
                nominal=1.0,
                lower=0.95,
                upper=1.05,
                unit="m",
            ),
            InitialConditionSpec(
                name="v0",
                label="Initial velocity",
                nominal=0.0,
                lower=-0.05,
                upper=0.05,
                unit="m/s",
            ),
        ),
        state_labels=("x", "v"),
        phase_reference_frequency=(spring_mass_reference_frequency),
        phase_band_strategy="trajectory_tube",
        default_t_start=0.0,
        default_t_end=30.0,
        default_time_points=1000,
        default_n_samples=200,
        default_seed=42,
        description=(
            "Configure mass, stiffness, damping and uncertain "
            "initial conditions before running the ensemble."
        ),
    )


def build_nonlinear_pendulum_model() -> ModelSpec:
    """Build the nonlinear-pendulum catalogue entry."""

    return ModelSpec(
        key="nonlinear_pendulum",
        name="Nonlinear pendulum",
        derivative=nonlinear_pendulum,
        parameters=(
            ParameterSpec(
                name="g",
                label="Gravitational acceleration",
                nominal=9.81,
                lower=9.75,
                upper=9.87,
                unit="m/s²",
            ),
            ParameterSpec(
                name="L",
                label="Pendulum length",
                nominal=1.0,
                lower=0.95,
                upper=1.05,
                unit="m",
            ),
        ),
        initial_conditions=(
            InitialConditionSpec(
                name="theta0",
                label="Initial angle",
                nominal=1.0,
                lower=0.95,
                upper=1.05,
                unit="rad",
            ),
            InitialConditionSpec(
                name="omega0",
                label="Initial angular velocity",
                nominal=0.0,
                lower=-0.05,
                upper=0.05,
                unit="rad/s",
            ),
        ),
        state_labels=("θ", "ω"),
        phase_reference_frequency=(pendulum_reference_frequency),
        phase_band_strategy="angular_energy",
        default_t_start=0.0,
        default_t_end=30.0,
        default_time_points=1000,
        default_n_samples=200,
        default_seed=42,
        description=(
            "Configure gravity, pendulum length and uncertain "
            "initial conditions before running the ensemble."
        ),
    )

def build_harmonic_oscillator_model() -> ModelSpec:
    """Build the harmonic-oscillator catalogue entry."""

    return ModelSpec(
        key="harmonic_oscillator",
        name="Harmonic oscillator",
        derivative=harmonic_oscillator,
        parameters=(
            ParameterSpec(
                name="m",
                label="Mass",
                nominal=1.0,
                lower=0.95,
                upper=1.05,
                unit="kg",
            ),
            ParameterSpec(
                name="k",
                label="Spring constant",
                nominal=1.0,
                lower=0.95,
                upper=1.05,
                unit="N/m",
            ),
        ),
        initial_conditions=(
            InitialConditionSpec(
                name="x0",
                label="Initial position",
                nominal=1.0,
                lower=0.95,
                upper=1.05,
                unit="m",
            ),
            InitialConditionSpec(
                name="v0",
                label="Initial velocity",
                nominal=0.0,
                lower=-0.05,
                upper=0.05,
                unit="m/s",
            ),
        ),
        state_labels=("x", "v"),
        phase_reference_frequency=(spring_mass_reference_frequency),
        default_t_start=0.0,
        default_t_end=30.0,
        default_time_points=1000,
        default_n_samples=200,
        default_seed=42,
        description=(
            "Configure the uncertain parameter intervals before "
            "running the ensemble."
        ),
    )