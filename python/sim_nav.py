# ==============================================================================
#  sim_nav.py
#  Simulated navigation: random start → fixed goal.
#
#  A synthetic 2-D environment (rectangular room + one obstacle) is created.
#  The robot starts at a random pose, and a proportional bearing controller
#  drives it toward a fixed goal.  Gaussian noise is added to both the encoder
#  and the LiDAR, then the measurements are fed through the full hybrid SLAM
#  pipeline (odometry + EKF + ICP).
#
#  Three trajectories are compared on the final plot:
#    - Ground truth   (noiseless kinematic integration)
#    - Odometry only  (noisy dead-reckoning estimate)
#    - EKF + ICP      (SLAM-corrected estimate)
#
#  Usage:
#    python sim_nav.py                          # random seed, default goal
#    python sim_nav.py --goal 0.9 0.8          # world-frame goal (metres)
#    python sim_nav.py --seed 42               # reproducible run
#    python sim_nav.py --seed 42 --goal 0.7 -0.6
# ==============================================================================

import argparse
import math
import os
import random
import time

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

import hybrid_slam as hs
from hybrid_slam import (
    wrap_angle, WHEEL_BASE, ENCODER_SCALE,
    scan_to_xy, USE_GEOMETRIC_INIT,
)
from live_slam import Scan, LiveSLAM

# ==============================================================================
#  Simulation constants
# ==============================================================================

# Calibrated from test6: speed=62 ≈ 0.264 m/s → 0.0264 m per 100 ms step
SIM_SPEED_SCALE  = 0.0264 / 62.0   # metres per step per speed unit

ENCODER_NOISE    = 1.5              # std dev, encoder counts per step (slip noise)
LIDAR_NOISE_M    = 0.015            # std dev, LiDAR distance in metres
N_RAYS           = 50               # LiDAR rays per scan (matches hardware)
LIDAR_RANGE_M    = 3.5              # max range in sim

# Room dimensions: ±_HALF_W × ±_HALF_H metres (5 m × 5 m)
# Larger room ensures any (1–1.5 m) goal is reachable from any random start.
_HW, _HH = 2.5, 2.5

# Plain rectangular room — four walls only.
# A simple room still provides four distinct edges for ICP matching.
ROOM_WALLS = [
    ((-_HW, -_HH), ( _HW, -_HH)),   # south
    (( _HW, -_HH), ( _HW,  _HH)),   # east
    (( _HW,  _HH), (-_HW,  _HH)),   # north
    ((-_HW,  _HH), (-_HW, -_HH)),   # west
]

# Goal is always specified in the SLAM frame (robot-initial frame: 0,0,0 = start).
# Example: (1.0, 0.5) means 1.0 m ahead and 0.5 m to the left of the start heading.
DEFAULT_GOAL_SLAM = (1.0, 0.6)

# ==============================================================================
#  Environment: ray casting
# ==============================================================================

def _ray_segment_t(ox, oy, dx, dy, x1, y1, x2, y2):
    """
    Parametric intersection of ray (ox,oy)+(t)*(dx,dy) with
    segment (x1,y1)→(x2,y2).  Returns t > 0 or None.
    """
    denom = dx * (y2 - y1) - dy * (x2 - x1)
    if abs(denom) < 1e-10:
        return None
    t = ((x1 - ox) * (y2 - y1) - (y1 - oy) * (x2 - x1)) / denom
    s = ((x1 - ox) * dy          - (y1 - oy) * dx         ) / denom
    if t > 1e-6 and 0.0 <= s <= 1.0:
        return t
    return None


def ray_cast(ox, oy, angle_world, walls=ROOM_WALLS):
    """Nearest wall distance along angle_world (radians, CCW+)."""
    dx, dy = math.cos(angle_world), math.sin(angle_world)
    best = LIDAR_RANGE_M
    for (x1, y1), (x2, y2) in walls:
        t = _ray_segment_t(ox, oy, dx, dy, x1, y1, x2, y2)
        if t is not None and t < best:
            best = t
    return best


def scan_from_pose(rx, ry, rtheta, rng):
    """
    Simulate one LiDAR scan from true pose (rx, ry, rtheta).
    Returns (angles_deg, distances_mm) matching the real hardware format
    consumed by scan_to_xy():
        angle_deg  — clockwise from robot forward direction
        distance_mm — millimetres
    """
    angles_deg = [i * 360.0 / N_RAYS for i in range(N_RAYS)]
    distances_mm = []
    for a_deg in angles_deg:
        # Real LiDAR convention: CW positive → negate for CCW world frame
        a_world = rtheta - math.radians(a_deg)
        d_true = ray_cast(rx, ry, a_world)
        d_noisy = max(0.03, d_true + rng.gauss(0, LIDAR_NOISE_M))
        distances_mm.append(d_noisy * 1000.0)
    return angles_deg, distances_mm


# ==============================================================================
#  Spatial validity helpers
# ==============================================================================

def is_free(x, y, margin=0.25):
    """Return True if (x, y) is inside the room with the given wall margin."""
    return abs(x) <= _HW - margin and abs(y) <= _HH - margin


# ==============================================================================
#  Simulated robot (world-frame state, noisy sensor output)
# ==============================================================================

class SimRobot:
    def __init__(self, x, y, theta, rng):
        self.true_x     = x
        self.true_y     = y
        self.true_theta = theta
        self.encoder    = 0    # cumulative noisy integer count

    def step(self, speed, steering_deg, rng):
        """
        Advance one 100 ms timestep.  Returns a Scan object with
        noisy encoder + noisy LiDAR (angles in degrees, distances in mm).
        Steering convention matches compute_odometry_trajectory():
            positive steering_deg → positive dtheta (CCW / left turn).
        """
        ds = SIM_SPEED_SCALE * float(speed)
        dtheta = (ds / WHEEL_BASE) * math.tan(math.radians(steering_deg))

        # Integrate true pose (same Euler order as compute_odometry_trajectory)
        self.true_theta = wrap_angle(self.true_theta + dtheta)
        self.true_x    += ds * math.cos(self.true_theta)
        self.true_y    += ds * math.sin(self.true_theta)

        # Noisy encoder
        true_counts = ds / ENCODER_SCALE
        noisy_delta = true_counts + rng.gauss(0, ENCODER_NOISE)
        self.encoder += int(round(noisy_delta))

        # Noisy LiDAR from true pose
        angles_deg, dists_mm = scan_from_pose(
            self.true_x, self.true_y, self.true_theta, rng)

        return Scan(self.encoder, steering_deg, angles_deg, dists_mm)


# ==============================================================================
#  Proportional bearing controller (operates in SLAM frame)
# ==============================================================================

class ProportionalController:
    """
    Goal-seeking proportional controller.
    Position/heading inputs are in the SLAM frame (starts at 0,0,0).
    The goal is also in the SLAM frame.
    """
    def __init__(self, gx, gy, base_speed=60, Kp=3.0,
                 max_steer=22.0, goal_radius=0.12, slow_radius=0.35):
        self.gx, self.gy = gx, gy
        self.base_speed  = base_speed
        self.Kp          = Kp
        self.max_steer   = max_steer
        self.goal_radius = goal_radius
        self.slow_radius = slow_radius

    def compute(self, x, y, theta):
        """Returns (speed_int, steering_deg_float)."""
        dx, dy = self.gx - x, self.gy - y
        dist   = math.hypot(dx, dy)
        if dist < self.goal_radius:
            return 0, 0.0
        bearing = math.atan2(dy, dx)
        herr    = wrap_angle(bearing - theta)
        steer   = max(-self.max_steer,
                      min( self.max_steer, self.Kp * math.degrees(herr)))
        speed = self.base_speed
        if dist < self.slow_radius:
            speed = max(28, int(self.base_speed * dist / self.slow_radius))
        return speed, steer

    def at_goal(self, x, y):
        return math.hypot(self.gx - x, self.gy - y) < self.goal_radius


# ==============================================================================
#  Frame transforms (world → SLAM/robot-initial frame)
# ==============================================================================

def true_poses_to_slam(poses_world, sx, sy, stheta):
    cs, ss = math.cos(-stheta), math.sin(-stheta)
    out = []
    for wx, wy, wth in poses_world:
        dx, dy = wx - sx, wy - sy
        rx = cs * dx - ss * dy
        ry = ss * dx + cs * dy
        out.append((rx, ry, wrap_angle(wth - stheta)))
    return out


def walls_to_slam(walls, sx, sy, stheta):
    cs, ss = math.cos(-stheta), math.sin(-stheta)
    out = []
    for (wx1, wy1), (wx2, wy2) in walls:
        dx1, dy1 = wx1 - sx, wy1 - sy
        dx2, dy2 = wx2 - sx, wy2 - sy
        rx1, ry1 = cs*dx1 - ss*dy1, ss*dx1 + cs*dy1
        rx2, ry2 = cs*dx2 - ss*dy2, ss*dx2 + cs*dy2
        out.append(((rx1, ry1), (rx2, ry2)))
    return out


# ==============================================================================
#  Simulation loop
# ==============================================================================

def run_simulation(gx_slam, gy_slam, seed, max_steps=500):
    """
    gx_slam, gy_slam  — goal in the SLAM frame (metres from start, along
                        start heading for X and left for Y).
    """
    rng = random.Random(seed)
    np.random.seed(seed)

    # ── Sample valid random start ─────────────────────────────────────────────
    # Ensure start is at least (goal_dist + 0.4 m) from every wall so the
    # straight-line path to the goal stays inside the room.
    goal_dist  = math.hypot(gx_slam, gy_slam)
    start_margin = min(goal_dist + 0.4, _HW * 0.80)   # cap at 80% of half-width
    sx = sy = None
    for _ in range(5000):
        cx = rng.uniform(-_HW + start_margin, _HW - start_margin)
        cy = rng.uniform(-_HH + start_margin, _HH - start_margin)
        if abs(cx) <= _HW - start_margin and abs(cy) <= _HH - start_margin:
            sx, sy = cx, cy
            break
    if sx is None:
        sx, sy = 0.0, 0.0  # fallback: centre
    stheta = rng.uniform(-math.pi, math.pi)

    print(f"[sim] World start : ({sx:.3f}, {sy:.3f}, {math.degrees(stheta):.1f}°)")
    print(f"[sim] SLAM goal   : ({gx_slam:.3f}, {gy_slam:.3f})")
    print(f"[sim] Seed        : {seed}")

    robot  = SimRobot(sx, sy, stheta, rng)
    ctrl   = ProportionalController(gx_slam, gy_slam)
    slam   = LiveSLAM()

    true_poses_world = [(sx, sy, stheta)]
    session_log      = []
    goal_reached     = False

    for step in range(max_steps):
        # Control: use EKF estimate once SLAM has initialised
        if slam.scan_count > 0:
            xe, ye, te = slam.ekf.state
        else:
            xe, ye, te = 0.0, 0.0, 0.0

        speed, steer = ctrl.compute(xe, ye, te)
        scan = robot.step(speed, steer, rng)
        slam.ingest(scan)
        session_log.append(scan)
        true_poses_world.append((robot.true_x, robot.true_y, robot.true_theta))

        # Stop when EKF estimate says we've reached the goal
        if ctrl.at_goal(xe, ye):
            ts_now = true_poses_to_slam(true_poses_world, sx, sy, stheta)
            true_dist_goal = math.hypot(ts_now[-1][0] - gx_slam, ts_now[-1][1] - gy_slam)
            print(f"[sim] EKF reached goal at step {step + 1}  "
                  f"(true dist to goal = {true_dist_goal:.3f} m)")
            goal_reached = True
            break

        # Collision guard on true position
        if not is_free(robot.true_x, robot.true_y, margin=0.06):
            print(f"[sim] Wall contact at step {step + 1} — stopping.")
            break
    else:
        print(f"[sim] Max steps ({max_steps}) reached without reaching goal.")

    return session_log, true_poses_world, sx, sy, stheta, gx_slam, gy_slam, goal_reached


# ==============================================================================
#  Post-run plots
# ==============================================================================

def plot_and_save(session_log, true_poses_world,
                  sx, sy, stheta, gx_slam, gy_slam, save_dir, seed, goal_reached=False):

    os.makedirs(save_dir, exist_ok=True)

    # Transform ground truth to SLAM frame
    true_slam = true_poses_to_slam(true_poses_world, sx, sy, stheta)
    gt_x  = [p[0] for p in true_slam]
    gt_y  = [p[1] for p in true_slam]
    gt_th = [p[2] for p in true_slam]

    # Odometry from pipeline
    odo_x, odo_y, odo_th = hs.compute_odometry_trajectory(session_log)

    # EKF + ICP from pipeline
    icp_meas, _, _, _    = hs.compute_icp_trajectory(session_log)
    ekf_x, ekf_y, ekf_th = hs.run_ekf_icp(session_log, icp_meas)

    n_icp = sum(1 for m in icp_meas if m is not None)
    n = min(len(gt_x), len(odo_x), len(ekf_x))
    t_axis = np.arange(n) * 0.1   # seconds

    odo_err = [math.hypot(gt_x[i] - odo_x[i], gt_y[i] - odo_y[i]) for i in range(n)]
    ekf_err = [math.hypot(gt_x[i] - ekf_x[i], gt_y[i] - ekf_y[i]) for i in range(n)]

    final_odo_err = odo_err[n - 1]
    final_ekf_err = ekf_err[n - 1]
    gt_goal_err   = math.hypot(gt_x[n-1] - gx_slam, gt_y[n-1] - gy_slam)

    walls_slam = walls_to_slam(ROOM_WALLS, sx, sy, stheta)

    # ── Figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(15, 9))
    status = 'GOAL REACHED' if goal_reached else 'stopped early'
    fig.suptitle(
        f"Simulation Navigation  |  seed={seed}  |  {status}  |  "
        f"USE_GEOMETRIC_INIT={USE_GEOMETRIC_INIT}  |  "
        f"ICP pairs accepted: {n_icp}\n"
        f"World start: ({sx:.2f}, {sy:.2f}, {math.degrees(stheta):.0f}°)   "
        f"SLAM goal: ({gx_slam:.2f}, {gy_slam:.2f})   "
        f"Final odo err: {final_odo_err:.3f} m   "
        f"Final EKF err: {final_ekf_err:.3f} m   "
        f"True dist to goal: {gt_goal_err:.3f} m",
        fontsize=10
    )
    gs = gridspec.GridSpec(2, 3, figure=fig,
                           left=0.06, right=0.97, top=0.87, bottom=0.08,
                           wspace=0.38, hspace=0.48)

    # Panel 1: Trajectory comparison (SLAM frame, spans 2 rows)
    ax1 = fig.add_subplot(gs[:, :2])
    for (p1, p2) in walls_slam:
        ax1.plot([p1[0], p2[0]], [p1[1], p2[1]], 'k-', lw=1.5, alpha=0.45)
    ax1.plot(gt_x[:n],  gt_y[:n],  'g-',  lw=2.2, label='Ground truth',  zorder=4)
    ax1.plot(odo_x[:n], odo_y[:n], 'b--', lw=1.6, label='Odometry',       zorder=3)
    ax1.plot(ekf_x[:n], ekf_y[:n], 'r-',  lw=1.8, label='EKF + ICP',      zorder=4)
    ax1.plot(0, 0, 'ko', ms=11, zorder=7, label='Start (SLAM origin)')
    ax1.plot(gx_slam, gy_slam, 'y*', ms=20, zorder=8, label='Goal')
    # End markers
    for xv, yv, col, mk in [(gt_x[n-1],  gt_y[n-1],  'green', '^'),
                              (odo_x[n-1], odo_y[n-1], 'blue',  '^'),
                              (ekf_x[n-1], ekf_y[n-1], 'red',   '^')]:
        ax1.plot(xv, yv, mk, color=col, ms=9, zorder=6)
    # Heading arrows at final pose
    for xv, yv, tv, col in [(gt_x[n-1],  gt_y[n-1],  gt_th[n-1],  'green'),
                              (ekf_x[n-1], ekf_y[n-1], ekf_th[n-1], 'red')]:
        ax1.annotate('',
                     xy=(xv + 0.12*math.cos(tv), yv + 0.12*math.sin(tv)),
                     xytext=(xv, yv),
                     arrowprops=dict(arrowstyle='->', color=col, lw=2.2))
    ax1.set_title('Trajectory comparison (robot-initial / SLAM frame)')
    ax1.set_xlabel('X (m)'); ax1.set_ylabel('Y (m)')
    ax1.set_aspect('equal'); ax1.grid(True, ls='--', alpha=0.45)
    ax1.legend(fontsize=9, loc='upper left')

    # Panel 2: Position error over time
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.plot(t_axis, odo_err[:n], 'b--', lw=1.5, label=f'Odo (final={final_odo_err:.3f} m)')
    ax2.plot(t_axis, ekf_err[:n], 'r-',  lw=1.8, label=f'EKF (final={final_ekf_err:.3f} m)')
    ax2.set_title('Position error vs ground truth')
    ax2.set_xlabel('Time (s)'); ax2.set_ylabel('Error (m)')
    ax2.legend(fontsize=9); ax2.grid(True, ls='--', alpha=0.45)

    # Panel 3: Heading error over time
    th_odo_err_deg = [math.degrees(abs(wrap_angle(gt_th[i] - odo_th[i]))) for i in range(n)]
    th_ekf_err_deg = [math.degrees(abs(wrap_angle(gt_th[i] - ekf_th[i]))) for i in range(n)]
    ax3 = fig.add_subplot(gs[1, 2])
    ax3.plot(t_axis, th_odo_err_deg[:n], 'b--', lw=1.5, label='Odo θ err')
    ax3.plot(t_axis, th_ekf_err_deg[:n], 'r-',  lw=1.8, label='EKF θ err')
    ax3.set_title('Heading error vs ground truth')
    ax3.set_xlabel('Time (s)'); ax3.set_ylabel('Error (deg)')
    ax3.legend(fontsize=9); ax3.grid(True, ls='--', alpha=0.45)

    ts   = time.strftime("%d_%m_%y_%H_%M_%S")
    path = os.path.join(save_dir, f"sim_nav_seed{seed}_{ts}.png")
    fig.savefig(path, dpi=150)
    print(f"[sim] Figure saved → {path}")

    plt.show()


# ==============================================================================
#  Entry point
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Simulated navigation from random start to fixed goal.")
    parser.add_argument('--goal', nargs=2, type=float,
                        default=list(DEFAULT_GOAL_SLAM),
                        metavar=('X', 'Y'),
                        help='Goal in SLAM frame: X=ahead (m), Y=left (m). '
                             'Relative to wherever the robot starts.')
    parser.add_argument('--seed', type=int,
                        default=random.randint(0, 9999),
                        help='Random seed (default: random)')
    parser.add_argument('--steps', type=int, default=500,
                        help='Maximum simulation steps (default: 500)')
    args = parser.parse_args()

    gx_w, gy_w = args.goal

    result = run_simulation(gx_w, gy_w, args.seed, args.steps)
    session_log, true_poses_world, sx, sy, stheta, gx_slam, gy_slam, goal_reached = result

    if len(session_log) < 5:
        print("[sim] Too few steps — no plot generated.")
        return

    _here    = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(_here, '..', 'results', 'nav_sim')
    plot_and_save(session_log, true_poses_world,
                  sx, sy, stheta, gx_slam, gy_slam, save_dir, args.seed, goal_reached)


if __name__ == '__main__':
    main()
