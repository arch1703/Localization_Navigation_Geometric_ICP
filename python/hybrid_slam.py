# ==============================================================
#  hybrid_slam.py
#  EKF + ICP Hybrid Localization Pipeline
#
#  Corresponds to:
#    "Indoor SLAM application using geometric and ICP matching
#     methods based on line features" — Cho, Kim & Kim (2018)
#    Table 3 Row 7:  SVD-based ICP + EKF  (USE_GEOMETRIC_INIT=False)
#    Table 3 Row ~5: Geometric init + ICP + EKF (USE_GEOMETRIC_INIT=True)
#
#  Pipeline:
#    Phase 2 — Pure odometry trajectory
#    Phase 3 — ICP-chained absolute trajectory
#    Phase 4 — EKF: odometry prediction + ICP correction
#    Phase 5 — Trajectory comparison plot (odometry / ICP / EKF+ICP)
#    Phase 6 — Geometric environment map
# ==============================================================

import math
import pickle
import random

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

# ── Toggle ─────────────────────────────────────────────────────────────────────
# Set True  → geometric line-matching initializes ICP  (paper Row 5 approach)
# Set False → odometry initializes ICP                 (paper Row 7 approach)
USE_GEOMETRIC_INIT = True

# ── Data file ──────────────────────────────────────────────────────────────────
# Path is relative to the project root (Localiztion_proj_code(arnav)/).
# Run this script from that directory, or adjust the path below.
import os as _os
_HERE    = _os.path.dirname(_os.path.abspath(__file__))          # python/
_DATADIR = _os.path.join(_HERE, "..", "data")                    # data/
FILENAME = _os.path.join(_DATADIR,
           "test6_robot_data_smooth_right(2)_62_6_04_05_26_01_02_33.pkl")

# ── Constants (from project handoff) ──────────────────────────────────────────
WHEEL_RADIUS           = 0.034         # metres
ENCODER_COUNTS_PER_REV = 152
WHEEL_BASE             = 0.15          # metres
# SLIP_FACTOR: calibrated from 10 straight runs + 5 diagonal runs at speed 100.
# Joint optimisation with STEER_SCALE gives RMSE = 4.2 cm across 5 diagonal runs.
SLIP_FACTOR            = 0.220
ENCODER_SCALE          = 2 * math.pi * WHEEL_RADIUS / ENCODER_COUNTS_PER_REV * SLIP_FACTOR  # m/count
# STEER_SCALE: servo linkage delivers only 45% of commanded angle to the wheels.
STEER_SCALE            = 0.400

ICP_STRIDE             = 4             # actual-index gap between ICP scan pairs
ICP_ERROR_THRESHOLD    = 0.10          # reject ICP pairs above this mean error (m)
MIN_SCAN_POINTS        = 20            # minimum LiDAR points for a valid scan
MIN_DS                 = 0.02          # minimum motion per pair (m) — skip near-stationary pairs
CONSISTENCY_THRESH     = math.radians(25)   # reject ICP rotation if delta vs odometry > this

# ==============================================================================
#  Section 1 — Shared helpers
#  (functions copied from icp_two_scans.py / plot_odometry_trajectory.py;
#   those files are NOT modified)
# ==============================================================================

def wrap_angle(theta):
    return math.atan2(math.sin(theta), math.cos(theta))


def encoder_counts_to_distance(delta_counts):
    return ENCODER_SCALE * delta_counts


def scan_to_xy(scan):
    """Convert a RobotSensorSignal scan to an (N,2) numpy array of XY points."""
    points = []
    for angle_deg, distance_mm in zip(scan.angles, scan.distances):
        if distance_mm <= 20:
            continue
        distance_m = distance_mm / 1000.0
        angle_rad  = -math.radians(angle_deg)
        x = distance_m * math.cos(angle_rad)
        y = distance_m * math.sin(angle_rad)
        if abs(x) < 5 and abs(y) < 5:
            points.append([x, y])
    return np.array(points) if points else np.empty((0, 2))


def rotate_points(points, theta):
    R = np.array([[math.cos(theta), -math.sin(theta)],
                  [math.sin(theta),  math.cos(theta)]])
    return points @ R.T


def icp(source, target, max_iters=20, tolerance=1e-5, max_match_distance=0.4):
    """
    SVD-based point-to-point ICP.
    Returns (aligned_source, R_total, t_total, mean_error).
    R_total, t_total map source → target:  R @ p + t ≈ p_target
    """
    src = source.copy()
    dst = target.copy()

    prev_error = float("inf")
    total_R    = np.eye(2)
    total_t    = np.zeros(2)

    for _ in range(max_iters):
        tree      = cKDTree(dst)
        distances, indices = tree.query(src)
        mask      = distances < max_match_distance

        if np.sum(mask) < 5:
            break

        src_valid   = src[mask]
        matched_dst = dst[indices[mask]]

        src_centroid = np.mean(src_valid,   axis=0)
        dst_centroid = np.mean(matched_dst, axis=0)
        H = (src_valid - src_centroid).T @ (matched_dst - dst_centroid)

        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        t   = dst_centroid - R @ src_centroid
        src = (R @ src.T).T + t

        total_R = R @ total_R
        total_t = R @ total_t + t

        mean_error = np.mean(distances[mask])
        if abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error

    return src, total_R, total_t, prev_error


# ==============================================================================
#  Section 1b — Geometric line extraction & matching
#  (only used when USE_GEOMETRIC_INIT = True)
# ==============================================================================

def extract_lines(points, n_iters=150, dist_thresh=0.05, min_inliers=12):
    """
    Extract wall-like line segments from a 2D point cloud using iterative RANSAC.

    Returns a list of (angle_rad, centroid_xy, inlier_array).
      angle_rad is in [0, pi) — the orientation of the line, not its direction.
    """
    if len(points) < min_inliers:
        return []

    remaining = points.copy()
    lines     = []

    while len(remaining) >= min_inliers:
        best_count  = 0
        best_normal = None
        best_c      = 0.0
        best_angle  = 0.0
        n           = len(remaining)

        for _ in range(n_iters):
            idx = np.random.choice(n, 2, replace=False)
            p1, p2 = remaining[idx[0]], remaining[idx[1]]
            dp     = p2 - p1
            length = np.linalg.norm(dp)
            if length < 1e-6:
                continue

            # Normal form:  nx*x + ny*y = c
            nx, ny = -dp[1] / length, dp[0] / length
            c      = nx * p1[0] + ny * p1[1]

            dists = np.abs(remaining @ np.array([nx, ny]) - c)
            count = int(np.sum(dists < dist_thresh))

            if count > best_count:
                best_count  = count
                best_normal = np.array([nx, ny])
                best_c      = c
                best_angle  = math.atan2(dp[1], dp[0]) % math.pi

        if best_count < min_inliers or best_normal is None:
            break

        dists       = np.abs(remaining @ best_normal - best_c)
        inlier_mask = dists < dist_thresh
        inliers     = remaining[inlier_mask]
        centroid    = np.mean(inliers, axis=0)
        lines.append((best_angle, centroid, inliers))
        remaining   = remaining[~inlier_mask]

    return lines


def geometric_align(lines_src, lines_dst, angle_thresh=0.175):
    """
    Match lines between two scans by angle similarity (modulo pi, within angle_thresh rad).
    Estimates relative rotation and translation from matched line pairs.

    Returns (dtheta, dx, dy, success).
    success=False → caller falls back to odometry initialization.

    Paper reference: Cho et al. 2018 — geometric matching before ICP refinement.
    """
    if len(lines_src) < 1 or len(lines_dst) < 1:
        return 0.0, 0.0, 0.0, False

    matches  = []
    used_dst = set()

    for i, (a_src, _, _) in enumerate(lines_src):
        best_diff = angle_thresh
        best_j    = -1
        for j, (a_dst, _, _) in enumerate(lines_dst):
            if j in used_dst:
                continue
            diff = abs(a_src - a_dst) % math.pi
            diff = min(diff, math.pi - diff)   # modulo-pi angular distance
            if diff < best_diff:
                best_diff = diff
                best_j    = j
        if best_j >= 0:
            matches.append((i, best_j))
            used_dst.add(best_j)

    if len(matches) < 2:
        return 0.0, 0.0, 0.0, False

    # Rotation: median of per-pair angle differences
    dtheta_samples = []
    for i, j in matches:
        diff = lines_dst[j][0] - lines_src[i][0]
        # Wrap to (-pi/2, pi/2) — lines have no direction, only orientation
        diff = (diff + math.pi / 2) % math.pi - math.pi / 2
        dtheta_samples.append(diff)

    dtheta = float(np.median(dtheta_samples))

    R_mat = np.array([[math.cos(dtheta), -math.sin(dtheta)],
                      [math.sin(dtheta),  math.cos(dtheta)]])

    # Translation: median of per-pair centroid displacements
    translations = []
    for i, j in matches:
        c_src = lines_src[i][1]
        c_dst = lines_dst[j][1]
        translations.append(c_dst - R_mat @ c_src)

    t_median = np.median(translations, axis=0)
    dx, dy   = float(t_median[0]), float(t_median[1])

    return dtheta, dx, dy, True


# ==============================================================================
#  Phase 2 — Pure odometry trajectory
# ==============================================================================

def compute_odometry_trajectory(robot_list):
    """
    Bicycle kinematic model using encoder counts and steering servo angle.
    Returns (x_list, y_list, theta_list) over all timesteps.
    """
    x, y, theta  = 0.0, 0.0, 0.0
    x_list       = [x]
    y_list       = [y]
    theta_list   = [theta]
    last_encoder = robot_list[0].encoder_counts

    for i in range(1, len(robot_list)):
        encoder      = robot_list[i].encoder_counts
        steering_deg = robot_list[i].steering
        delta_counts = encoder - last_encoder
        last_encoder = encoder

        ds           = encoder_counts_to_distance(delta_counts)
        steering_rad = math.radians(steering_deg * STEER_SCALE)
        dtheta       = (ds / WHEEL_BASE) * math.tan(steering_rad)

        theta = wrap_angle(theta + dtheta)
        x    += ds * math.cos(theta)
        y    += ds * math.sin(theta)

        x_list.append(x)
        y_list.append(y)
        theta_list.append(theta)

    return x_list, y_list, theta_list


# ==============================================================================
#  Phase 3 — ICP-chained absolute trajectory
# ==============================================================================

def compute_icp_trajectory(robot_list, stride=ICP_STRIDE):
    """
    Pair consecutive scans stride indices apart, compute ICP between each pair,
    and chain the transforms into an absolute trajectory.

    ICP initialization is controlled by USE_GEOMETRIC_INIT:
      True  → geometric line matching (paper Row 5 approach)
      False → odometry initial guess  (paper Row 7 approach)
    Falls back to odometry if geometric matching fails.

    Returns:
      icp_measurements : dict {timestep_idx: (x, y, theta, icp_error)}
                         used by the EKF for correction steps
      icp_x_list, icp_y_list, icp_theta_list : trajectory for plotting
    """
    icp_measurements = {}
    icp_x_list       = [0.0]
    icp_y_list       = [0.0]
    icp_theta_list   = [0.0]

    x, y, theta = 0.0, 0.0, 0.0

    for i in range(0, len(robot_list) - stride, stride):
        j = i + stride

        if robot_list[i].num_lidar_rays < MIN_SCAN_POINTS:
            continue
        if robot_list[j].num_lidar_rays < MIN_SCAN_POINTS:
            continue

        scan_i = scan_to_xy(robot_list[i])
        scan_j = scan_to_xy(robot_list[j])

        if len(scan_i) < MIN_SCAN_POINTS or len(scan_j) < MIN_SCAN_POINTS:
            continue

        # ── Odometry between i and j ────────────────────────────────────────
        encoder_diff = robot_list[j].encoder_counts - robot_list[i].encoder_counts
        ds           = encoder_counts_to_distance(encoder_diff)
        steering_rad = math.radians(robot_list[i].steering * STEER_SCALE)
        dtheta_odom  = (ds / WHEEL_BASE) * math.tan(steering_rad)

        # ── Minimum motion filter ────────────────────────────────────────────
        if abs(ds) < MIN_DS:
            continue

        # ── ICP initialization (toggle-controlled) ──────────────────────────
        # Physical scan-to-scan transform: scan_j = R(-dtheta_robot) @ scan_i + [-ds, 0]
        # So the correct initial guess uses -dtheta_odom and -ds.
        # geometric_align returns the scan-to-scan transform directly (positive signs).
        if USE_GEOMETRIC_INIT:
            lines_i = extract_lines(scan_i)
            lines_j = extract_lines(scan_j)
            dtheta_init, dx_init, dy_init, geo_ok = geometric_align(lines_i, lines_j)
            if not geo_ok:
                # Fallback: physically correct odometry convention
                dtheta_init, dx_init, dy_init = -dtheta_odom, -ds, 0.0
        else:
            # Physically correct odometry convention:
            # scan_j ≈ R(-dtheta_odom) @ scan_i + [-ds, 0]
            dtheta_init, dx_init, dy_init = -dtheta_odom, -ds, 0.0

        scan_i_init = rotate_points(scan_i, dtheta_init) + np.array([dx_init, dy_init])

        # ── Run ICP ─────────────────────────────────────────────────────────
        _, R_icp, t_icp, icp_error = icp(scan_i_init, scan_j)

        if icp_error > ICP_ERROR_THRESHOLD:
            continue

        # ── Compose total scan-to-scan transform ─────────────────────────────
        # R_total, t_total satisfy: scan_j ≈ R_total @ scan_i + t_total
        # Physical meaning: R_total = R(-dtheta_robot),  t_total ≈ [-ds_robot, 0]
        R_init_mat = np.array([[math.cos(dtheta_init), -math.sin(dtheta_init)],
                                [math.sin(dtheta_init),  math.cos(dtheta_init)]])
        R_total = R_icp @ R_init_mat
        t_total = R_icp @ np.array([dx_init, dy_init]) + t_icp

        # Extract physical robot motion:
        #   R_total = R(-dtheta_robot)  →  dtheta_robot = -atan2(R_total[1,0], R_total[0,0])
        #   t_total ≈ [-ds_robot, 0]   →  ds_robot ≈ -t_total[0] in local frame j
        dtheta_physical = -math.atan2(R_total[1, 0], R_total[0, 0])

        # ── Consistency check (catches 180° ICP flips) ───────────────────────
        if abs(wrap_angle(dtheta_physical - dtheta_odom)) > CONSISTENCY_THRESH:
            continue

        # ── Chain to global pose ─────────────────────────────────────────────
        # Robot displacement in global frame: R(theta_new) @ (-t_total)
        # where -t_total ≈ [ds_robot, 0] in the local frame j
        theta_new    = wrap_angle(theta + dtheta_physical)
        R_global_new = np.array([[math.cos(theta_new), -math.sin(theta_new)],
                                  [math.sin(theta_new),  math.cos(theta_new)]])
        delta_pos = R_global_new @ (-t_total)
        x_new     = x + delta_pos[0]
        y_new     = y + delta_pos[1]

        x, y, theta = x_new, y_new, theta_new

        icp_measurements[j] = (x, y, theta, icp_error)
        icp_x_list.append(x)
        icp_y_list.append(y)
        icp_theta_list.append(theta)

    return icp_measurements, icp_x_list, icp_y_list, icp_theta_list


# ==============================================================================
#  Phase 4 — EKF: bicycle-model prediction + ICP measurement correction
# ==============================================================================

class EKF:
    """
    3-state EKF  x = [x, y, theta]  with bicycle kinematic motion model.

    Prediction : odometry (encoder delta counts + servo steering angle)
    Correction : ICP-derived absolute pose [x_icp, y_icp, theta_icp]

    Process noise R_u scales with motion magnitude.
    Measurement noise Q scales adaptively with ICP mean error (Cho et al. 2018 ER term).
    """

    I3 = np.eye(3)

    def __init__(self, x0=None, sigma0=None):
        self.state = np.array(x0 if x0 is not None else [0.0, 0.0, 0.0], dtype=float)
        self.cov   = sigma0 if sigma0 is not None else np.eye(3) * 1e-6

    def predict(self, delta_counts, steering_deg):
        ds    = encoder_counts_to_distance(delta_counts)
        phi   = math.radians(steering_deg * STEER_SCALE)
        L     = WHEEL_BASE
        theta = self.state[2]

        dtheta    = (ds / L) * math.tan(phi)
        theta_mid = theta + dtheta / 2.0

        # Predicted state (bicycle kinematics)
        x_pred = np.array([
            self.state[0] + ds * math.cos(theta_mid),
            self.state[1] + ds * math.sin(theta_mid),
            wrap_angle(theta + dtheta)
        ])

        # Jacobian  G_x = df/dx  (3×3)
        G_x = np.array([
            [1, 0, -ds * math.sin(theta_mid)],
            [0, 1,  ds * math.cos(theta_mid)],
            [0, 0,  1]
        ])

        # Jacobian  G_u = df/du,  u = [ds, phi]  (3×2)
        cos2_phi = math.cos(phi) ** 2
        G_u = np.array([
            [math.cos(theta_mid),  0.0],
            [math.sin(theta_mid),  0.0],
            [math.tan(phi) / L,    ds / (L * cos2_phi) if cos2_phi > 1e-9 else 0.0]
        ])

        # Process noise R_u (2×2) — scales with motion magnitude
        var_s   = max(1e-6, 0.30 * ds  ** 2 + 1e-4)
        var_phi = max(1e-6, 0.15 * phi ** 2 + 1e-3)
        R_u = np.diag([var_s, var_phi])

        self.state = x_pred
        self.cov   = G_x @ self.cov @ G_x.T + G_u @ R_u @ G_u.T

    def correct(self, z_icp, icp_error):
        """
        z_icp     : [x_icp, y_icp, theta_icp] — absolute pose from ICP chain
        icp_error : mean ICP residual (m) — used to scale measurement noise Q.

        H = I3 (measurement directly observes the state).
        Q scales adaptively: high ICP error → wide Q → low correction weight.
        """
        H = self.I3.copy()

        # Adaptive measurement noise (Cho et al. 2018 ER concept)
        e = max(icp_error, 1e-4)
        Q = np.diag([e, e, e * 0.1])

        z          = np.array(z_icp, dtype=float)
        innovation = z - self.state
        innovation[2] = wrap_angle(innovation[2])   # wrap angle difference

        S = H @ self.cov @ H.T + Q

        # Mahalanobis distance gating — reject measurements >4.5σ from prediction
        # (chi-squared 3 DOF: 95% = 7.81, 99% = 11.34; use 20 for a loose gate)
        mahal_sq = float(innovation @ np.linalg.inv(S) @ innovation)
        if mahal_sq > 20.0:
            return  # reject inconsistent ICP measurement

        K = self.cov @ H.T @ np.linalg.inv(S)

        self.state    = self.state + K @ innovation
        self.state[2] = wrap_angle(self.state[2])
        self.cov      = (self.I3 - K @ H) @ self.cov


def run_ekf_icp(robot_list, icp_measurements):
    """
    Runs the full EKF over every timestep.
      - Predicts with odometry at every step.
      - Corrects with ICP measurement when one is available for that timestep.

    icp_measurements : dict {timestep_idx: (x, y, theta, icp_error)}

    Returns (ekf_x, ekf_y, ekf_theta) lists over all timesteps.
    """
    ekf = EKF(x0=[0.0, 0.0, 0.0], sigma0=np.eye(3) * 1e-6)

    ekf_x_list     = [ekf.state[0]]
    ekf_y_list     = [ekf.state[1]]
    ekf_theta_list = [ekf.state[2]]

    last_encoder = robot_list[0].encoder_counts

    for i in range(1, len(robot_list)):
        encoder      = robot_list[i].encoder_counts
        steering_deg = robot_list[i].steering
        delta_counts = encoder - last_encoder
        last_encoder = encoder

        # Prediction step (every timestep)
        ekf.predict(delta_counts, steering_deg)

        # Correction step (only when an ICP measurement exists)
        if i in icp_measurements:
            x_icp, y_icp, theta_icp, icp_error = icp_measurements[i]
            ekf.correct([x_icp, y_icp, theta_icp], icp_error)

        ekf_x_list.append(ekf.state[0])
        ekf_y_list.append(ekf.state[1])
        ekf_theta_list.append(ekf.state[2])

    return ekf_x_list, ekf_y_list, ekf_theta_list


# ==============================================================================
#  Phase 5 — Trajectory comparison plot
# ==============================================================================

def plot_trajectory_comparison(odo_x, odo_y, icp_x, icp_y, ekf_x, ekf_y):
    init_label = "Geometric init" if USE_GEOMETRIC_INIT else "Odometry init"
    row_ref    = "5" if USE_GEOMETRIC_INIT else "7"

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.plot(odo_x, odo_y, 'b--', linewidth=1.5, label='Odometry only')
    ax.plot(icp_x, icp_y, color='orange', linewidth=1.5, label='ICP-chained')
    ax.plot(ekf_x, ekf_y, 'g-',  linewidth=2.0, label='EKF + ICP')

    ax.plot(odo_x[0],  odo_y[0],  'ko',  markersize=8,  label='Start')
    ax.plot(odo_x[-1], odo_y[-1], 'b^',  markersize=8,  label='End (odometry)')
    ax.plot(ekf_x[-1], ekf_y[-1], 'g^',  markersize=8,  label='End (EKF+ICP)')

    ax.set_title(
        f"Trajectory Comparison — {init_label}\n"
        f"(Cho et al. 2018, Table 3 Row {row_ref} approach)"
    )
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_aspect("equal")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend()
    fig.tight_layout()
    plt.show()


# ==============================================================================
#  Phase 6 — Geometric environment map
# ==============================================================================

def build_global_map(robot_list, ekf_poses_full):
    """
    Transforms each valid LiDAR scan into the global frame using the EKF pose
    at that timestep, then accumulates all points into a single global map.

    ekf_poses_full : list of (x, y, theta), one per timestep.
    Returns an (N, 2) array of global-frame map points.
    """
    global_points = []
    for i, scan in enumerate(robot_list):
        if scan.num_lidar_rays < MIN_SCAN_POINTS:
            continue
        local_pts = scan_to_xy(scan)
        if len(local_pts) == 0:
            continue
        x, y, theta = ekf_poses_full[i]
        R = np.array([[math.cos(theta), -math.sin(theta)],
                      [math.sin(theta),  math.cos(theta)]])
        global_pts = (R @ local_pts.T).T + np.array([x, y])
        global_points.append(global_pts)

    return np.vstack(global_points) if global_points else np.empty((0, 2))


def plot_global_map(global_pts, ekf_x, ekf_y):
    init_label = "Geometric init" if USE_GEOMETRIC_INIT else "Odometry init"

    fig, ax = plt.subplots(figsize=(9, 7))
    if len(global_pts) > 0:
        ax.scatter(global_pts[:, 0], global_pts[:, 1],
                   s=1, c="gray", alpha=0.3, label="LiDAR map points")
    ax.plot(ekf_x, ekf_y, "g-", linewidth=2, label="EKF + ICP trajectory")
    ax.plot(ekf_x[0], ekf_y[0], "ko", markersize=8, label="Start")

    ax.set_title(f"Geometric Environment Map — {init_label}")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_aspect("equal")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend()
    fig.tight_layout()
    plt.show()


# ==============================================================================
#  Main
# ==============================================================================

def main():
    print("=" * 60)
    print("  Hybrid SLAM: EKF + ICP Localization Pipeline")
    print(f"  USE_GEOMETRIC_INIT = {USE_GEOMETRIC_INIT}")
    print("=" * 60)

    # Load data
    print(f"\nLoading: {FILENAME}")
    with open(FILENAME, "rb") as f:
        data = pickle.load(f)
    robot_list = data["robot_sensor_signal"]
    print(f"  {len(robot_list)} timesteps loaded.")

    # ── Phase 2: Odometry ─────────────────────────────────────────────────────
    print("\n[Phase 2] Odometry trajectory...")
    odo_x, odo_y, odo_theta = compute_odometry_trajectory(robot_list)
    print(f"  Final odometry pose: "
          f"({odo_x[-1]:.3f} m, {odo_y[-1]:.3f} m, "
          f"{math.degrees(odo_theta[-1]):.1f}°)")

    # ── Phase 3: ICP-chained ──────────────────────────────────────────────────
    init_label = "geometric" if USE_GEOMETRIC_INIT else "odometry"
    print(f"\n[Phase 3] ICP trajectory (stride={ICP_STRIDE}, init={init_label})...")
    icp_measurements, icp_x, icp_y, icp_theta = compute_icp_trajectory(robot_list)
    print(f"  {len(icp_measurements)} ICP correction poses generated.")
    if icp_x:
        print(f"  Final ICP pose: ({icp_x[-1]:.3f} m, {icp_y[-1]:.3f} m, "
              f"{math.degrees(icp_theta[-1]):.1f}°)")

    # ── Phase 4: EKF ─────────────────────────────────────────────────────────
    print("\n[Phase 4] EKF with ICP corrections...")
    ekf_x, ekf_y, ekf_theta = run_ekf_icp(robot_list, icp_measurements)
    print(f"  Final EKF pose: "
          f"({ekf_x[-1]:.3f} m, {ekf_y[-1]:.3f} m, "
          f"{math.degrees(ekf_theta[-1]):.1f}°)")

    # ── Phase 5: Comparison plot ──────────────────────────────────────────────
    print("\n[Phase 5] Trajectory comparison plot...")
    plot_trajectory_comparison(odo_x, odo_y, icp_x, icp_y, ekf_x, ekf_y)

    # ── Phase 6: Geometric map ────────────────────────────────────────────────
    print("\n[Phase 6] Building geometric map...")
    ekf_poses_full = list(zip(ekf_x, ekf_y, ekf_theta))
    global_pts     = build_global_map(robot_list, ekf_poses_full)
    print(f"  Map contains {len(global_pts)} points.")
    plot_global_map(global_pts, ekf_x, ekf_y)

    print("\nDone.")


if __name__ == "__main__":
    main()
