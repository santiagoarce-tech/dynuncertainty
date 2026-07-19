import numpy as np
import matplotlib.pyplot as plt
from dynuncertainty.phase_envelope import (
    break_wrapped_angle_discontinuities,
    wrap_angles_to_principal_interval,
)


def plot_parameter_bounds(parameter_intervals, nominal_values=None, title="Parameter uncertainty bounds"):
    """
    Plot parameter uncertainty bounds.

    Parameters
    ----------
    parameter_intervals : dict
        Dictionary of parameter intervals.

        Example:
            {
                "m": (0.95, 1.05),
                "k": (0.95, 1.05),
                "x0": (0.95, 1.05),
            }

    nominal_values : dict, optional
        Dictionary of nominal parameter values.

        Example:
            {
                "m": 1.0,
                "k": 1.0,
                "x0": 1.0,
            }

    title : str
        Plot title.
    """
    names = list(parameter_intervals.keys())
    y_positions = np.arange(len(names))

    lows = np.array([parameter_intervals[name][0] for name in names])
    highs = np.array([parameter_intervals[name][1] for name in names])
    centers = 0.5 * (lows + highs)
    half_widths = 0.5 * (highs - lows)

    plt.figure(figsize=(8, 0.7 * len(names) + 2))

    plt.errorbar(
        centers,
        y_positions,
        xerr=half_widths,
        fmt="o",
        capsize=6,
        label="uncertainty interval",
    )

    if nominal_values is not None:
        nominal_x = [nominal_values.get(name, np.nan) for name in names]

        plt.scatter(
            nominal_x,
            y_positions,
            marker="x",
            s=80,
            label="nominal value",
        )

    for i, name in enumerate(names):
        low = lows[i]
        high = highs[i]

        plt.text(
            low,
            y_positions[i] + 0.18,
            f"{low:.4g}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

        plt.text(
            high,
            y_positions[i] + 0.18,
            f"{high:.4g}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.yticks(y_positions, names)
    plt.xlabel("parameter value")
    plt.title(title)
    plt.grid(True, axis="x", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_temporal_results(
    times,
    lower_position,
    median_position,
    upper_position,
    nominal_position,
    model_name,
    configuration_text,
    state_label="x",
    target_axis=None,
):
    """
    Plot temporal uncertainty results for one state component.

    Parameters
    ----------
    times
        Simulation times.
    lower_position
        Pointwise lower percentile of the ensemble.
    median_position
        Pointwise ensemble median.
    upper_position
        Pointwise upper percentile of the ensemble.
    nominal_position
        Position along the nominal trajectory.
    model_name
        Human-readable model name.
    configuration_text
        Formatted parameter and initial-condition summary.

    Returns
    -------
    matplotlib.figure.Figure
        Completed temporal-results figure.
    """

    creates_own_figure = target_axis is None

    if creates_own_figure:
        figure, axis = plt.subplots(
            figsize=(9, 6)
        )
    else:
        axis = target_axis
        figure = axis.figure

    axis.fill_between(
        times,
        lower_position,
        upper_position,
        alpha=0.30,
        label="5%-95% band",
    )

    axis.plot(
        times,
        median_position,
        linewidth=2,
        label=f"pointwise median {state_label}(t)",
    )

    axis.plot(
        times,
        nominal_position,
        color="black",
        linestyle="--",
        linewidth=1.8,
        label=f"nominal {state_label}(t)",
    )

    axis.set_title(
    f"{model_name}: uncertainty in {state_label}(t)"
    )

    axis.set_xlabel("time")
    axis.set_ylabel(state_label)
    axis.grid(True)

    axis.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 0.50),
        borderaxespad=0.0,
        title="Legend",
    )


    if creates_own_figure:
        figure.subplots_adjust(right=0.70)

    return figure

def plot_time_resolved_phase_tube(
    axis,
    centre_positions,
    centre_velocities,
    tube_radii,
    reference_frequency,
):
    """
    Draw a continuous empirical tube along the phase-space evolution.

    The tube follows the time-ordered ensemble centres. Its lateral
    boundaries are obtained by offsetting the centre curve along its
    local normal direction using empirical Monte Carlo radii.
    """

    centre_positions = np.asarray(
        centre_positions,
        dtype=float,
    )
    centre_velocities = np.asarray(
        centre_velocities,
        dtype=float,
    )
    tube_radii = np.asarray(
        tube_radii,
        dtype=float,
    )

    tangent_positions = np.gradient(centre_positions)
    tangent_velocities = np.gradient(centre_velocities)

    tangent_norms = np.hypot(
        tangent_positions,
        tangent_velocities,
    )

    safe_tangent_norms = np.where(
        tangent_norms > 1.0e-12,
        tangent_norms,
        1.0,
    )

    normal_positions = (
        -tangent_velocities / safe_tangent_norms
    )
    normal_velocities = (
        tangent_positions / safe_tangent_norms
    )

    first_boundary_positions = (
        centre_positions
        + tube_radii * normal_positions
    )
    first_boundary_velocities = (
        centre_velocities
        + tube_radii * normal_velocities
    )

    second_boundary_positions = (
        centre_positions
        - tube_radii * normal_positions
    )
    second_boundary_velocities = (
        centre_velocities
        - tube_radii * normal_velocities
    )

    physical_first_velocities = (
        reference_frequency * first_boundary_velocities
    )
    physical_second_velocities = (
        reference_frequency * second_boundary_velocities
    )

    polygon_positions = np.concatenate(
        (
            first_boundary_positions,
            second_boundary_positions[::-1],
            first_boundary_positions[:1],
        )
    )

    polygon_velocities = np.concatenate(
        (
            physical_first_velocities,
            physical_second_velocities[::-1],
            physical_first_velocities[:1],
        )
    )

    axis.fill(
        polygon_positions,
        polygon_velocities,
        color="mediumpurple",
        alpha=0.13,
        label="Time-resolved 90% Monte Carlo tube",
        zorder=2,
    )

    axis.plot(
        first_boundary_positions,
        physical_first_velocities,
        color="darkviolet",
        linestyle="--",
        linewidth=1.2,
        label="Time-resolved tube boundaries",
        zorder=4,
    )

    axis.plot(
        second_boundary_positions,
        physical_second_velocities,
        color="darkviolet",
        linestyle="--",
        linewidth=1.2,
        label=None,
        zorder=4,
    )

def plot_pendulum_separatrix(
    axis,
    reference_frequency,
):
    """Draw the nominal separatrix of the nonlinear pendulum."""

    angular_coordinates = np.linspace(
        -np.pi,
        np.pi,
        600,
    )

    positive_separatrix = (
        2.0
        * reference_frequency
        * np.cos(0.5 * angular_coordinates)
    )

    negative_separatrix = -positive_separatrix

    axis.plot(
        angular_coordinates,
        positive_separatrix,
        color="darkorange",
        linestyle="-.",
        linewidth=1.8,
        label="Nominal pendulum separatrix",
        zorder=5,
    )

    axis.plot(
        angular_coordinates,
        negative_separatrix,
        color="darkorange",
        linestyle="-.",
        linewidth=1.8,
        label=None,
        zorder=5,
    )

def plot_phase_space_results(
    positions,
    velocities,
    nominal_trajectory,
    inner_band_positions,
    inner_band_velocities,
    outer_band_positions,
    outer_band_velocities,
    model_name,
    configuration_text,
    n_samples,
    state_labels=("x", "v"),
    target_axis=None,
    phase_band_strategy="radial",
    tube_centre_positions=None,
    tube_centre_velocities=None,
    tube_radii=None,
    reference_frequency=None,
    angular_phase_positions=None,
    angular_phase_velocities=None,
):
    """
    Plot a Monte Carlo phase-space ensemble and its radial band.

    Parameters
    ----------
    positions
        Position histories with shape (n_samples, n_times).
    velocities
        Velocity histories with shape (n_samples, n_times).
    nominal_trajectory
        Nominal trajectory with shape (n_times, state_dimension).
    inner_band_positions, inner_band_velocities
        Coordinates of the empirical inner radial boundary.
    outer_band_positions, outer_band_velocities
        Coordinates of the empirical outer radial boundary.
    model_name
        Human-readable model name.
    configuration_text
        Formatted parameter and initial-condition summary.
    n_samples
        Number of Monte Carlo trajectories.

    Returns
    -------
    matplotlib.figure.Figure
        Completed phase-space figure.
    """

    creates_own_figure = target_axis is None

    if creates_own_figure:
        figure, axis = plt.subplots(
            figsize=(8, 6)
        )
    else:
        axis = target_axis
        figure = axis.figure
    
    first_state_label, second_state_label = state_labels

    if phase_band_strategy == "angular_energy":
        if (
            angular_phase_positions is None
            or angular_phase_velocities is None
        ):
            raise ValueError(
                "The angular phase-space data are incomplete."
            )

        displayed_positions = angular_phase_positions
        displayed_velocities = angular_phase_velocities

    else:
        displayed_positions = positions
        displayed_velocities = velocities

    for trajectory_index, (position, velocity) in enumerate(
        zip(displayed_positions, displayed_velocities)
    ):
        axis.plot(
            position,
            velocity,
            color="steelblue",
            alpha=0.08,
            linewidth=0.8,
            label=(
                rf"Monte Carlo trajectories "
                rf"$\gamma_i(t)=("
                rf"{first_state_label}_i(t),"
                rf"{second_state_label}_i(t))$, "
                rf"$i=1,\ldots,{n_samples}$"
                if trajectory_index == 0
                else None
            ),
        )

    if phase_band_strategy == "radial":
        closed_inner_positions = np.append(
            inner_band_positions,
            inner_band_positions[0],
        )

        closed_inner_velocities = np.append(
            inner_band_velocities,
            inner_band_velocities[0],
        )

        closed_outer_positions = np.append(
            outer_band_positions,
            outer_band_positions[0],
        )

        closed_outer_velocities = np.append(
            outer_band_velocities,
            outer_band_velocities[0],
        )

        band_polygon_positions = np.concatenate(
            (
                outer_band_positions,
                inner_band_positions[::-1],
                outer_band_positions[:1],
            )
        )

        band_polygon_velocities = np.concatenate(
            (
                outer_band_velocities,
                inner_band_velocities[::-1],
                outer_band_velocities[:1],
            )
        )

        axis.fill(
            band_polygon_positions,
            band_polygon_velocities,
            color="mediumpurple",
            alpha=0.12,
            label="Empirical Monte Carlo radial band",
            zorder=1,
        )

        axis.plot(
            closed_outer_positions,
            closed_outer_velocities,
            color="darkviolet",
            linestyle="--",
            linewidth=1.8,
            label="Empirical Monte Carlo radial boundaries",
            zorder=5,
        )

        axis.plot(
            closed_inner_positions,
            closed_inner_velocities,
            color="darkviolet",
            linestyle="--",
            linewidth=1.8,
            label=None,
            zorder=5,
        )

    elif phase_band_strategy == "trajectory_tube":
        if (
            tube_centre_positions is None
            or tube_centre_velocities is None
            or tube_radii is None
            or reference_frequency is None
        ):
            raise ValueError(
                "The trajectory-tube data are incomplete."
            )

        plot_time_resolved_phase_tube(
            axis=axis,
            centre_positions=tube_centre_positions,
            centre_velocities=tube_centre_velocities,
            tube_radii=tube_radii,
            reference_frequency=reference_frequency,
        )

    elif phase_band_strategy == "angular_energy":
        if reference_frequency is None:
            raise ValueError(
                "The angular-energy strategy requires a "
                "reference frequency."
            )

        plot_pendulum_separatrix(
            axis=axis,
            reference_frequency=reference_frequency,
        )

    else:
        raise ValueError(
            f"Unknown phase-band strategy: "
            f"{phase_band_strategy!r}."
        )
    

    if phase_band_strategy == "angular_energy":
        wrapped_nominal_positions = (
            wrap_angles_to_principal_interval(
                nominal_trajectory[:, 0]
            )
        )

        (
            displayed_nominal_positions,
            displayed_nominal_velocities,
        ) = break_wrapped_angle_discontinuities(
            wrapped_angles=wrapped_nominal_positions,
            companion_values=nominal_trajectory[:, 1],
        )

    else:
        displayed_nominal_positions = nominal_trajectory[:, 0]
        displayed_nominal_velocities = nominal_trajectory[:, 1]

    axis.plot(
        displayed_nominal_positions,
        displayed_nominal_velocities,
        color="black",
        linestyle="--",
        linewidth=2.0,
        label=r"Nominal trajectory $\gamma_{\mathrm{nom}}(t)$",
        zorder=3,
    )

    axis.set_title(
        f"{model_name}: phase-space ensemble"
    )
    axis.set_xlabel(first_state_label)
    axis.set_ylabel(second_state_label)
    axis.grid(True)

    axis.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 0.50),
        borderaxespad=0.0,
        title="Legend",
    )



    if creates_own_figure:
        figure.subplots_adjust(right=0.70)

    return figure


def plot_simulation_results(
    times,
    lower_position,
    median_position,
    upper_position,
    nominal_position,
    positions,
    velocities,
    nominal_trajectory,
    inner_band_positions,
    inner_band_velocities,
    outer_band_positions,
    outer_band_velocities,
    model_name,
    configuration_text,
    n_samples,
    state_label="x",
    phase_state_labels=("x", "v"),
    phase_band_strategy="radial",
    tube_centre_positions=None,
    tube_centre_velocities=None,
    tube_radii=None,
    reference_frequency=None,
    angular_phase_positions=None,
    angular_phase_velocities=None,
):
    """
    Create one result window containing both simulation views.

    The upper panel shows the temporal uncertainty band. The lower
    panel shows the phase-space ensemble and its empirical radial
    band.
    """

    figure, (time_axis, phase_axis) = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(14, 11),
    )

    window_manager = figure.canvas.manager

    if window_manager is not None:
        window_manager.set_window_title(
            f"dynuncertainty — {model_name} results"
        )

    plot_temporal_results(
        times=times,
        lower_position=lower_position,
        median_position=median_position,
        upper_position=upper_position,
        nominal_position=nominal_position,
        model_name=model_name,
        configuration_text=configuration_text,
        state_label=state_label,
        target_axis=time_axis,
    )

    plot_phase_space_results(
        positions=positions,
        velocities=velocities,
        nominal_trajectory=nominal_trajectory,
        inner_band_positions=inner_band_positions,
        inner_band_velocities=inner_band_velocities,
        outer_band_positions=outer_band_positions,
        outer_band_velocities=outer_band_velocities,
        model_name=model_name,
        configuration_text=configuration_text,
        n_samples=n_samples,
        state_labels=phase_state_labels,
        target_axis=phase_axis,
        phase_band_strategy=phase_band_strategy,
        tube_centre_positions=tube_centre_positions,
        tube_centre_velocities=tube_centre_velocities,
        tube_radii=tube_radii,
        reference_frequency=reference_frequency,
        angular_phase_positions=angular_phase_positions,
        angular_phase_velocities=angular_phase_velocities,
    )

    figure.text(
        0.73,
        0.50,
        "Configuration\n\n" + configuration_text,
        fontsize=9,
        verticalalignment="center",
        horizontalalignment="left",
        bbox={
            "boxstyle": "round,pad=0.7",
            "facecolor": "whitesmoke",
            "edgecolor": "gray",
            "alpha": 0.95,
        },
    )

    figure.subplots_adjust(
        right=0.70,
        hspace=0.38,
    )

    return figure