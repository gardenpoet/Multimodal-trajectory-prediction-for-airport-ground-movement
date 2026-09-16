import numpy as np
import torch
from geographiclib.geodesic import Geodesic
from math import sin, cos, radians
from torch import tensor
from typing import Tuple


def wrap_angle(angle):
    return np.radians(((angle % 360) + 540) % 360 - 180)


def transform_points_2d(points: np.array, ref_point: np.array, theta: float) -> np.array:
    """Transforms a set of input points following: P = (P - t) @ R.

    Inputs
    ------
        points[np.array]:    input points to be transformed.
        ref_point[np.array]: point used to translate the input points.
        theta[float]:        rotation angle.

    Outputs
    -------
        tf_points[np.array]: transformed points.
    """
    tf_points = points.copy()
    R = np.array(
        [[cos(theta), -sin(theta)],
         [sin(theta),  cos(theta)]])
    tf_points[:, 0:2] = np.matmul(points[:, 0:2] - ref_point.reshape(1, 2), R)
    tf_points[:, 2:4] = np.matmul(points[:, 2:4] - ref_point.reshape(1, 2), R)
    return tf_points


def inv_transform(traj_rel: np.array, start_abs: np.array, theta: float) -> np.array:
    """Transforms a relative trajectory back to absolute coordinates.

    Applies: P_abs = P_rel @ R(theta).T + start_abs

    Inputs
    ------
        traj_rel[np.array]:  relative trajectory, shape (N, T, 2).
        start_abs[np.array]: absolute start position, shape (2,).
        theta[float]:        heading angle in degrees.

    Outputs
    -------
        tf_points[np.array]: absolute trajectory, same shape as traj_rel.
    """
    heading = radians(theta)
    R = np.array(
        [[cos(heading), -sin(heading)],
         [sin(heading),  cos(heading)]])
    rot_coords = traj_rel @ R.T
    return rot_coords + start_abs


def inv_transform_batch(
    traj_rel: np.ndarray,   # (B, T, 2)
    start_abs: np.ndarray,  # (B, 2)
    theta: np.ndarray,      # (B,)
) -> np.ndarray:
    """Batched version of inv_transform.

    Applies a per-sample rotation and translation to B trajectories in one
    vectorised numpy operation, replacing B sequential inv_transform calls.

        P_abs[b] = P_rel[b] @ R(theta[b]).T + start_abs[b]

    Inputs
    ------
        traj_rel[np.ndarray]:  relative trajectories, shape (B, T, 2).
        start_abs[np.ndarray]: absolute start positions, shape (B, 2).
        theta[np.ndarray]:     heading angles in degrees, shape (B,).

    Outputs
    -------
        np.ndarray: absolute trajectories, shape (B, T, 2).
    """
    headings = np.radians(theta)                 # (B,)
    cos_h    = np.cos(headings)                  # (B,)
    sin_h    = np.sin(headings)                  # (B,)

    # Build one rotation matrix per sample: (B, 2, 2)
    R = np.stack([
        np.stack([ cos_h, -sin_h], axis=-1),    # first row:  [cos, -sin]
        np.stack([ sin_h,  cos_h], axis=-1),    # second row: [sin,  cos]
    ], axis=1)                                   # (B, 2, 2)

    # Apply per-sample rotation via einsum:
    #   rot_coords[b, t, j] = sum_i traj_rel[b, t, i] * R[b, j, i]
    # which is equivalent to traj_rel[b] @ R[b].T for each b.
    rot_coords = np.einsum('bti,bji->btj', traj_rel, R)   # (B, T, 2)

    # Translate by the per-sample start position
    return rot_coords + start_abs[:, None, :]              # (B, T, 2)


def direct_wrapper(geodesic, b, r, ref_lat, ref_lon, r_scale):
    """Computes lat/lon from range (r) and bearing (b) for a single sample.

    Inputs
    ------
        geodesic:        Geodesic instance (WGS84).
        b[array-like]:   bearing in degrees, shape (T,).
        r[array-like]:   range values, shape (T,).
        ref_lat[float]:  reference latitude.
        ref_lon[float]:  reference longitude.
        r_scale[float]:  scale factor applied to range before geodesic solve.

    Returns
    -------
        lat_array[list]: latitudes for each point.
        lon_array[list]: longitudes for each point.
    """
    lat_array = []
    lon_array = []
    for i in range(r.shape[0]):
        g = geodesic.Direct(ref_lat, ref_lon, b[i], r[i] * r_scale)
        lat_array.append(g['lat2'])
        lon_array.append(g['lon2'])
    return lat_array, lon_array


def direct_wrapper_batch(
    geodesic,
    b: np.ndarray,      # (B, T)
    r: np.ndarray,      # (B, T)
    ref_lat: float,
    ref_lon: float,
    r_scale: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Batched version of direct_wrapper.

    Flattens (B, T) inputs into (B*T,), runs the geodesic loop once, then
    reshapes results back to (B, T).  The geodesic.Direct C-extension cannot
    be vectorised, so the inner loop is unavoidable; however, calling this
    function once instead of B times eliminates B-1 Python function-call
    round-trips and allows the caller to stay free of sample-level loops.

    When all B samples share the same reference point (the common case when
    a batch comes from a single airport), this should be called once with the
    shared ref_lat / ref_lon.  When samples span multiple airports, fall back
    to calling direct_wrapper per sample.

    Inputs
    ------
        geodesic:        Geodesic instance (WGS84).
        b[np.ndarray]:   bearing in degrees, shape (B, T).
        r[np.ndarray]:   range values, shape (B, T).
        ref_lat[float]:  reference latitude  (shared across the batch).
        ref_lon[float]:  reference longitude (shared across the batch).
        r_scale[float]:  scale factor applied to range before geodesic solve.

    Returns
    -------
        lats[np.ndarray]: latitudes,  shape (B, T).
        lons[np.ndarray]: longitudes, shape (B, T).
    """
    B, T    = b.shape
    b_flat  = b.ravel()                           # (B*T,)
    r_flat  = r.ravel()                           # (B*T,)

    lat_flat = np.empty(B * T, dtype=np.float64)
    lon_flat = np.empty(B * T, dtype=np.float64)

    # Single loop over B*T points - identical per-point cost to the original,
    # but avoids B separate direct_wrapper call frames.
    for i in range(B * T):
        g            = geodesic.Direct(ref_lat, ref_lon, b_flat[i], r_flat[i] * r_scale)
        lat_flat[i]  = g['lat2']
        lon_flat[i]  = g['lon2']

    return lat_flat.reshape(B, T), lon_flat.reshape(B, T)


def xy_to_ll(
    traj_rel: tensor,
    start_abs_xy: tensor,
    start_heading: tensor,
    reference: Tuple,
    geodesic: Geodesic,
    return_xyabs: bool = False
) -> np.array:
    """Converts a relative XY trajectory to lat/lon coordinates.

    Unchanged from the original - used by plot_scene_batch and any other
    single-sample caller.  evaluate_prediction uses inv_transform_batch and
    direct_wrapper_batch instead to avoid per-sample Python overhead.

    Inputs
    ------
        traj_rel[tensor]:     relative trajectory, shape (N, T, 2).
        start_abs_xy[tensor]: absolute start position in XY, shape (2,).
        start_heading[tensor]: heading at the start position (degrees).
        reference[Tuple]:     (ref_lat, ref_lon, range_scale).
        geodesic[Geodesic]:   WGS84 geodesic instance.
        return_xyabs[bool]:   if True, also return absolute XY trajectory.

    Returns
    -------
        traj_ll[tensor]: trajectory in lat/lon, shape (N, T, 2).
        traj_xy_abs[tensor]: absolute XY trajectory (only if return_xyabs).
    """
    N, _, _ = traj_rel.shape
    traj_ll = torch.zeros_like(traj_rel)
    traj_xy_abs = inv_transform(traj_rel.cpu().numpy(), start_abs_xy, start_heading)
    for n in range(N):
        x, y = traj_xy_abs[n, :, 0], traj_xy_abs[n, :, 1]
        rang    = np.sqrt(x ** 2 + y ** 2)
        bearing = np.degrees(np.arctan2(y, x))
        lat, lon = direct_wrapper(
            geodesic, bearing, rang,
            reference[0], reference[1], reference[2]
        )
        traj_ll[n, :, 1] = torch.tensor(lon)
        traj_ll[n, :, 0] = torch.tensor(lat)
    if return_xyabs:
        return traj_ll, torch.tensor(traj_xy_abs)
    return traj_ll