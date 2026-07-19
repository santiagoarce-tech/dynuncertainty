"""
Geometric tools for empirical phase-space envelopes.

This module contains transformations and calculations used to
describe the region explored by Monte Carlo trajectories in
phase space.

The resulting regions are empirical geometric summaries. They
must not be interpreted as rigorous reachable sets or confidence
regions.
"""

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


def normalize_phase_coordinates(
    positions: FloatArray,
    velocities: FloatArray,
    reference_frequency: float,
) -> tuple[FloatArray, FloatArray]:
    """
    Express position and velocity using comparable physical units.

    For an oscillator with reference angular frequency ``omega_ref``,
    the scaled phase coordinates are

        X = x
        V = v / omega_ref

    Therefore, X and V both have units of position.

    Parameters
    ----------
    positions
        Position values from one or more trajectories.
    velocities
        Velocity values with the same shape as ``positions``.
    reference_frequency
        Positive reference angular frequency.

    Returns
    -------
    tuple of numpy.ndarray
        Scaled position and velocity arrays, preserving the original
        array shape.
    """

    position_array = np.asarray(positions, dtype=float)
    velocity_array = np.asarray(velocities, dtype=float)

    if position_array.shape != velocity_array.shape:
        raise ValueError(
            "Positions and velocities must have the same shape."
        )

    if not np.isfinite(reference_frequency):
        raise ValueError(
            "The reference frequency must be finite."
        )

    if reference_frequency <= 0.0:
        raise ValueError(
            "The reference frequency must be positive."
        )

    normalized_positions = position_array
    normalized_velocities = (
        velocity_array / reference_frequency
    )

    return normalized_positions, normalized_velocities

def compute_phase_polar_coordinates(
    normalized_positions: FloatArray,
    normalized_velocities: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    """
    Convert normalized phase coordinates into polar coordinates.

    Parameters
    ----------
    normalized_positions
        Position coordinates expressed in the chosen reference scale.
    normalized_velocities
        Velocity coordinates expressed in the same physical units as
        ``normalized_positions``.

    Returns
    -------
    tuple of numpy.ndarray
        Radii and angles, preserving the original array shape.
        Angles are expressed in radians within the interval
        [-pi, pi].
    """

    position_array = np.asarray(
        normalized_positions,
        dtype=float,
    )

    velocity_array = np.asarray(
        normalized_velocities,
        dtype=float,
    )

    if position_array.shape != velocity_array.shape:
        raise ValueError(
            "Normalized positions and velocities must have "
            "the same shape."
        )

    if not np.all(np.isfinite(position_array)):
        raise ValueError(
            "Normalized positions must contain only finite values."
        )

    if not np.all(np.isfinite(velocity_array)):
        raise ValueError(
            "Normalized velocities must contain only finite values."
        )

    radii = np.hypot(
        position_array,
        velocity_array,
    )

    angles = np.arctan2(
        velocity_array,
        position_array,
    )

    return radii, angles

def compute_radial_phase_band(
    radii: FloatArray,
    angles: FloatArray,
    n_angular_bins: int = 360,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """
    Estimate an empirical radial phase-space band.

    The angular interval [-pi, pi] is divided into equal sectors.
    For each sector, the minimum and maximum sampled radii are
    computed.

    Parameters
    ----------
    radii
        Radial coordinates of the sampled phase-space points.
    angles
        Angular coordinates, in radians, with the same shape as
        ``radii``.
    n_angular_bins
        Number of equal angular sectors.

    Returns
    -------
    tuple of numpy.ndarray
        Angular bin centers, inner radii and outer radii.

    Notes
    -----
    This function describes only the supplied Monte Carlo points.
    It does not calculate a rigorous reachable set or a confidence
    region.
    """

    radius_array = np.asarray(
        radii,
        dtype=float,
    ).ravel()

    angle_array = np.asarray(
        angles,
        dtype=float,
    ).ravel()

    if radius_array.shape != angle_array.shape:
        raise ValueError(
            "Radii and angles must contain the same number "
            "of values."
        )

    if radius_array.size == 0:
        raise ValueError(
            "At least one phase-space point is required."
        )

    if not np.all(np.isfinite(radius_array)):
        raise ValueError(
            "Radii must contain only finite values."
        )

    if not np.all(np.isfinite(angle_array)):
        raise ValueError(
            "Angles must contain only finite values."
        )

    if np.any(radius_array < 0.0):
        raise ValueError(
            "Radii cannot be negative."
        )

    if n_angular_bins < 4:
        raise ValueError(
            "At least four angular bins are required."
        )

    bin_edges = np.linspace(
        -np.pi,
        np.pi,
        n_angular_bins + 1,
    )

    bin_centers = 0.5 * (
        bin_edges[:-1] + bin_edges[1:]
    )

    bin_indices = np.digitize(
        angle_array,
        bin_edges,
        right=False,
    ) - 1

    bin_indices = np.clip(
        bin_indices,
        0,
        n_angular_bins - 1,
    )

    inner_radii = np.full(
        n_angular_bins,
        np.nan,
        dtype=float,
    )

    outer_radii = np.full(
        n_angular_bins,
        np.nan,
        dtype=float,
    )

    for bin_index in range(n_angular_bins):
        points_in_bin = radius_array[
            bin_indices == bin_index
        ]

        if points_in_bin.size > 0:
            inner_radii[bin_index] = np.min(
                points_in_bin
            )

            outer_radii[bin_index] = np.max(
                points_in_bin
            )

    if np.any(np.isnan(inner_radii)):
        raise ValueError(
            "Some angular sectors contain no phase-space points. "
            "Use fewer angular bins or provide more samples."
        )

    return bin_centers, inner_radii, outer_radii

def polar_boundary_to_phase_coordinates(
    angles: FloatArray,
    radii: FloatArray,
    reference_frequency: float,
) -> tuple[FloatArray, FloatArray]:
    """
    Convert a normalized polar boundary back to physical phase space.

    The inverse transformation is

        x = r cos(phi)
        v = omega_ref r sin(phi)

    Parameters
    ----------
    angles
        Boundary angles in radians.
    radii
        Boundary radii with the same shape as ``angles``.
    reference_frequency
        Positive reference angular frequency.

    Returns
    -------
    tuple of numpy.ndarray
        Physical position and velocity coordinates.
    """

    angle_array = np.asarray(
        angles,
        dtype=float,
    )

    radius_array = np.asarray(
        radii,
        dtype=float,
    )

    if angle_array.shape != radius_array.shape:
        raise ValueError(
            "Angles and radii must have the same shape."
        )

    if not np.all(np.isfinite(angle_array)):
        raise ValueError(
            "Angles must contain only finite values."
        )

    if not np.all(np.isfinite(radius_array)):
        raise ValueError(
            "Radii must contain only finite values."
        )

    if np.any(radius_array < 0.0):
        raise ValueError(
            "Radii cannot be negative."
        )

    if not np.isfinite(reference_frequency):
        raise ValueError(
            "The reference frequency must be finite."
        )

    if reference_frequency <= 0.0:
        raise ValueError(
            "The reference frequency must be positive."
        )

    positions = radius_array * np.cos(angle_array)

    velocities = (
        reference_frequency
        * radius_array
        * np.sin(angle_array)
    )

    return positions, velocities

def compute_time_resolved_phase_tube(
    normalized_positions: FloatArray,
    normalized_velocities: FloatArray,
    coverage_percentile: float = 90.0,
    n_time_slices: int = 60,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """
    Estimate time-resolved cross-sections of a phase-space tube.

    At selected times, the ensemble centre is estimated componentwise
    by the median. The tube radius is the requested percentile of the
    normalized distances from that centre.

    The result is an empirical Monte Carlo summary, not a confidence
    region or a rigorous reachable set.
    """

    position_array = np.asarray(
        normalized_positions,
        dtype=float,
    )
    velocity_array = np.asarray(
        normalized_velocities,
        dtype=float,
    )

    if position_array.shape != velocity_array.shape:
        raise ValueError(
            "Normalized positions and velocities must have "
            "the same shape."
        )

    if position_array.ndim != 2:
        raise ValueError(
            "A phase tube requires arrays with shape "
            "(n_samples, n_times)."
        )

    if not 0.0 < coverage_percentile <= 100.0:
        raise ValueError(
            "The coverage percentile must lie in (0, 100]."
        )

    if n_time_slices < 2:
        raise ValueError(
            "At least two time slices are required."
        )

    n_times = position_array.shape[1]

    selected_indices = np.unique(
        np.linspace(
            0,
            n_times - 1,
            min(n_time_slices, n_times),
            dtype=int,
        )
    )

    selected_positions = position_array[:, selected_indices]
    selected_velocities = velocity_array[:, selected_indices]

    centre_positions = np.median(
        selected_positions,
        axis=0,
    )
    centre_velocities = np.median(
        selected_velocities,
        axis=0,
    )

    radial_distances = np.hypot(
        selected_positions - centre_positions,
        selected_velocities - centre_velocities,
    )

    tube_radii = np.percentile(
        radial_distances,
        coverage_percentile,
        axis=0,
    )

    return centre_positions, centre_velocities, tube_radii

def wrap_angles_to_principal_interval(
    angles: FloatArray,
) -> FloatArray:
    """
    Wrap angular coordinates into the interval [-pi, pi).

    Angles differing by an integer multiple of 2*pi represent the
    same physical orientation of the pendulum.
    """

    angle_array = np.asarray(
        angles,
        dtype=float,
    )

    if not np.all(np.isfinite(angle_array)):
        raise ValueError(
            "Angles must contain only finite values."
        )

    wrapped_angles = (
        (angle_array + np.pi)
        % (2.0 * np.pi)
        - np.pi
    )

    return wrapped_angles    

def break_wrapped_angle_discontinuities(
    wrapped_angles: FloatArray,
    companion_values: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    """
    Break plotted curves where wrapped angles cross the branch cut.

    NaN values are inserted immediately after jumps larger than pi,
    preventing Matplotlib from drawing false lines across the complete
    angular interval.
    """

    angle_array = np.asarray(
        wrapped_angles,
        dtype=float,
    )

    companion_array = np.asarray(
        companion_values,
        dtype=float,
    )

    if angle_array.shape != companion_array.shape:
        raise ValueError(
            "Wrapped angles and companion values must have "
            "the same shape."
        )

    if angle_array.ndim < 1:
        raise ValueError(
            "Wrapped-angle trajectories require at least one axis."
        )

    if not np.all(np.isfinite(angle_array)):
        raise ValueError(
            "Wrapped angles must contain only finite values."
        )

    if not np.all(np.isfinite(companion_array)):
        raise ValueError(
            "Companion values must contain only finite values."
        )

    discontinuities = np.zeros(
        angle_array.shape,
        dtype=bool,
    )

    discontinuities[..., 1:] = (
        np.abs(np.diff(angle_array, axis=-1)) > np.pi
    )

    plotted_angles = angle_array.copy()
    plotted_companion_values = companion_array.copy()

    plotted_angles[discontinuities] = np.nan
    plotted_companion_values[discontinuities] = np.nan

    return plotted_angles, plotted_companion_values

def compute_pendulum_energy_ratio(
    initial_angles: FloatArray,
    initial_angular_velocities: FloatArray,
    gravity_values: FloatArray,
    length_values: FloatArray,
) -> FloatArray:
    """
    Compute the initial pendulum energy relative to the separatrix.

    The energy is expressed per unit bob mass. A ratio below one
    indicates libration, while a ratio above one indicates rotation.
    """

    (
        angle_array,
        angular_velocity_array,
        gravity_array,
        length_array,
    ) = np.broadcast_arrays(
        np.asarray(initial_angles, dtype=float),
        np.asarray(initial_angular_velocities, dtype=float),
        np.asarray(gravity_values, dtype=float),
        np.asarray(length_values, dtype=float),
    )

    if not np.all(np.isfinite(angle_array)):
        raise ValueError(
            "Initial angles must contain only finite values."
        )

    if not np.all(np.isfinite(angular_velocity_array)):
        raise ValueError(
            "Initial angular velocities must contain only "
            "finite values."
        )

    if not np.all(np.isfinite(gravity_array)):
        raise ValueError(
            "Gravity values must contain only finite values."
        )

    if not np.all(np.isfinite(length_array)):
        raise ValueError(
            "Pendulum lengths must contain only finite values."
        )

    if np.any(gravity_array <= 0.0):
        raise ValueError(
            "Gravity values must be positive."
        )

    if np.any(length_array <= 0.0):
        raise ValueError(
            "Pendulum lengths must be positive."
        )

    specific_energy = (
        0.5
        * length_array**2
        * angular_velocity_array**2
        + gravity_array
        * length_array
        * (1.0 - np.cos(angle_array))
    )

    separatrix_energy = (
        2.0
        * gravity_array
        * length_array
    )

    return specific_energy / separatrix_energy