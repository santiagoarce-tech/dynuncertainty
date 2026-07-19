# Dynuncertainty — Mark 1

`dynuncertainty` is a Python application for studying how uncertainty in model parameters and initial conditions propagates through dynamical systems.

Given intervals for the uncertain quantities, the program generates a Monte Carlo ensemble, numerically integrates every trajectory, and displays the resulting temporal and phase-space behavior together with the nominal solution.

Mark 1 provides three physically validated models:

- Harmonic oscillator;
- damped oscillator;
- nonlinear pendulum.

## Installation

Open a terminal in the project root, create a virtual environment, and install the package in editable mode:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

## Included models

### 1. Harmonic oscillator

The harmonic oscillator solves

`m x'' + kx = 0`

using position `x` and velocity `v` as state variables. The application displays the 5–95% temporal uncertainty band, ensemble median, nominal trajectory, and phase-space behavior. Conservation of nominal mechanical energy is verified automatically.

### 2. Damped oscillator

The damped oscillator solves

`m x'' + c x' + kx = 0`

using position `x` and velocity `v`. Its mechanical energy must decrease according to

`dE/dt = -c v² <= 0`.

The application verifies this dissipative behavior automatically and provides three phase-tube smoothness levels: Low, Medium, and High.

### 3. Nonlinear pendulum

The nonlinear pendulum solves

`theta'' + (g/L) sin(theta) = 0`

using angular position and angular velocity. Angles are wrapped into `[-pi, pi]` for phase-space visualization.

The nominal separatrix and the energy ratio `rho = E/E_sep` are used to classify the motion as:

- libration;
- near separatrix;
- positive rotation;
- negative rotation.

Rotation direction is confirmed from the temporal angular displacement, and conservation of nominal mechanical energy is verified automatically.

## Phase-space uncertainty strategies

Each model uses a phase-space representation adapted to its physical structure:

- `radial` — used by the harmonic oscillator to represent the radial spread of the ensemble around the nominal phase-space motion;
- `trajectory_tube` — used by the damped oscillator to construct an uncertainty tube around its decaying phase trajectory;
- `angular_energy` — used by the nonlinear pendulum to combine wrapped angular coordinates with the system's energy structure and nominal separatrix.

For `trajectory_tube`, the Low, Medium, and High smoothness settings modify the visual refinement of the tube without changing the underlying simulated trajectories.

## Numerical integration

Mark 1 uses the explicit Runge–Kutta method `DOP853` with the following tolerances:

- relative tolerance: `rtol = 1e-8`;
- absolute tolerance: `atol = 1e-10`.

Production simulations use chunked ensemble integration with `chunk_size = 20`. The ensemble is divided into blocks of at most 20 trajectories, which substantially improves performance while preserving the accuracy required by the physical validation checks.

The sequential implementation remains available as the scientific reference, while the full collective implementation is retained for research and benchmarking.

## Running the application

After activating the virtual environment and completing the editable installation, the application can be launched with either command:

```bash
python -m dynuncertainty
```

or through the installed console command:

```bash
dynuncertainty
```

Both commands open the model catalog, from which the harmonic oscillator, damped oscillator, or nonlinear pendulum interface can be selected.

## Operational limits

To prevent excessive memory consumption, Mark 1 validates every configuration before allocating the simulation arrays.

The maximum accepted values are:

- simulation duration: `90 s`;
- time points: `10,000`;
- Monte Carlo samples: `20,000`;
- combined computational load: `samples * time_points <= 20,000,000`.

A configuration must satisfy all four conditions. Values within the individual limits may still be rejected if their combined load exceeds the permitted maximum.

These limits protect the computer from configurations that would require several gigabytes of memory or make the graphical interface unresponsive.

## Rendering performance

Every calculated ensemble trajectory is drawn in the figures. Mark 1 does not visually subsample or discard trajectories when the number of Monte Carlo samples increases.

This makes the effect of increasing the ensemble size directly observable. However, for large ensembles, rendering all trajectories may become slower than the numerical integration itself and can temporarily make the graphical interface less responsive.

The operational limits protect memory usage, but they do not guarantee instantaneous rendering for every accepted configuration.

## Running the tests

From the project root, with the virtual environment activated, run:

```bash
python -m unittest discover -s tests -v
```

The Mark 1 test suite verifies:

- energy conservation for the harmonic oscillator;
- monotonic energy dissipation for the damped oscillator;
- energy conservation for the nonlinear pendulum;
- numerical agreement between chunked production integration and sequential reference integration.

For an ensemble of `N` samples, Euclidean division gives `N = 20q + r`, where `0 <= r < 20`. The production implementation processes `q` complete blocks of 20 trajectories and, when `r > 0`, one final block containing the remaining `r` trajectories. The current ensemble test uses `N = 25`, so it verifies both a complete block and a nonempty remainder block.

## Mark 1 status and scope

Mark 1 is a complete and functional foundation for the dynamic propagation of parameter and initial-condition uncertainty.

Its current scope includes:

- interval-based uncertain inputs;
- Monte Carlo ensemble generation;
- numerical integration of all sampled trajectories;
- temporal uncertainty bands;
- model-specific phase-space uncertainty representations;
- nominal physical validation through energy behavior;
- graphical configuration and visualization;
- computational safety limits;
- automated scientific and structural tests.

Mark 1 supports the three models documented above. It does not yet provide a general custom-equation editor, external model import, formal sensitivity analysis, parameter estimation, or experimental-data calibration.

The application has been developed and tested primarily on Linux Mint. Packaging, a system launcher, and a dedicated application icon are handled as separate distribution steps.