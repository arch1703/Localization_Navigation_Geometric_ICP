# ==============================================================================
#  real_nav.py
#  Real hardware navigation: drive to a user-specified goal from wherever
#  the robot currently is (treated as the SLAM origin).
#
#  A proportional bearing controller uses the EKF+ICP position estimate to
#  steer the robot toward the goal.  At the end the script generates a
#  comparison plot:
#    - Odometry trajectory  ("predicted" dead-reckoning path)
#    - EKF + ICP trajectory ("actual best estimate" SLAM-corrected path)
#    - Final position error between the two
#
#  Safety:
#    Ctrl-C → immediate e-stop + save
#    --timeout  : hard stop after N seconds  (default 60 s)
#    --estop-m  : hard stop if SLAM distance from origin > N metres (default 5 m)
#
#  Usage:
#    python real_nav.py --goal 1.5 0.0         # drive 1.5 m forward
#    python real_nav.py --goal 1.0 1.0         # diagonal goal
#    python real_nav.py --goal 0.8 -0.5 --speed 55
#    python real_nav.py --goal 1.2 0.3 --timeout 90
# ==============================================================================

import argparse
import math
import os
import pickle
import time

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

import hybrid_slam as hs
from hybrid_slam import (
    wrap_angle, scan_to_xy, USE_GEOMETRIC_INIT,
    encoder_counts_to_distance,
)
from live_slam import (
    Scan, LiveSLAM,
    make_socket, recv_sensor, send_stop,
    ARDUINO_IP, PORT, SEND_EVERY,
    LIVE_ICP_STRIDE,
)
from hybrid_slam import MIN_SCAN_POINTS

# ==============================================================================
#  Constants
# ==============================================================================

DEFAULT_BASE_SPEED  = 60     # 0–100 integer
DEFAULT_TIMEOUT_S   = 60     # seconds before forced stop
DEFAULT_ESTOP_M     = 5.0    # stop if odometry distance from origin exceeds this
GOAL_RADIUS_M       = 0.05   # metres — within this = "at goal"
SLOW_RADIUS_M       = 0.15   # metres — begin slowing here
MIN_SPEED           = 45     # minimum speed to keep the robot rolling
KP_STEER            = 2.5    # proportional gain (degrees of steering per degree heading error)
MAX_STEER_DEG       = 22.0   # steering saturation limit
PLOT_INTERVAL_S     = 0.20   # minimum seconds between live plot refreshes

# ==============================================================================
#  Proportional bearing controller  (SLAM-frame inputs)
# ==============================================================================

class ProportionalController:
    """
    Steers toward (gx, gy) using the robot's current SLAM pose estimate.
    All coordinates are in the SLAM frame (origin = robot start position).
    """
    def __init__(self, gx, gy, base_speed=DEFAULT_BASE_SPEED):
        self.gx, self.gy  = gx, gy
        self.base_speed   = base_speed

    def compute(self, x, y, theta):
        """Returns (speed_int, steering_deg_float)."""
        dx, dy = self.gx - x, self.gy - y
        dist   = math.hypot(dx, dy)
        if dist < GOAL_RADIUS_M:
            return 0, 0.0
        bearing = math.atan2(dy, dx)
        herr    = wrap_angle(bearing - theta)
        steer   = max(-MAX_STEER_DEG,
                      min( MAX_STEER_DEG, KP_STEER * math.degrees(herr)))
        speed = self.base_speed
        if dist < SLOW_RADIUS_M:
            speed = max(MIN_SPEED, int(self.base_speed * dist / SLOW_RADIUS_M))
        return int(speed), steer

    def at_goal(self, x, y):
        return math.hypot(self.gx - x, self.gy - y) < GOAL_RADIUS_M


# ==============================================================================
#  UDP command sender
# ==============================================================================

_last_send = 0.0

def send_command(sock, speed: int, steering: float):
    global _last_send
    now = time.perf_counter()
    if now - _last_send >= SEND_EVERY:
        msg = f"{int(speed)}, {int(steering)}\n"
        sock.sendto(msg.encode(), (ARDUINO_IP, PORT))
        _last_send = now


# ==============================================================================
#  Live display (2-panel: trajectory + current scan)
# ==============================================================================

def _make_live_fig(goal_x, goal_y):
    fig = plt.figure(figsize=(11, 6))
    fig.canvas.manager.set_window_title("Real Navigation — live")
    ax_t = fig.add_subplot(1, 2, 1)
    ax_s = fig.add_subplot(1, 2, 2)

    ax_t.set_title("Live trajectory (SLAM frame)")
    ax_t.set_xlabel("X (m)"); ax_t.set_ylabel("Y (m)")
    ax_t.set_aspect('equal'); ax_t.grid(True, ls='--', alpha=0.4)
    ax_t.plot(0, 0, 'ko', ms=9, label='Start')
    ax_t.plot(goal_x, goal_y, 'y*', ms=18, label='Goal')
    line_odo, = ax_t.plot([], [], 'b--', lw=1.5, label='Odometry')
    line_ekf, = ax_t.plot([], [], 'r-',  lw=2.0, label='EKF + ICP')
    ax_t.legend(fontsize=9)
    status_txt = ax_t.text(0.02, 0.97, '', transform=ax_t.transAxes,
                            va='top', fontsize=8,
                            bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.8))

    ax_s.set_title("Current LiDAR scan"); ax_s.set_aspect('equal')
    ax_s.set_xlim(-4, 4); ax_s.set_ylim(-4, 4)
    ax_s.grid(True, ls='--', alpha=0.4)
    scat, = ax_s.plot([], [], '.', color='steelblue', ms=3, alpha=0.7)
    ax_s.plot(0, 0, 'ro', ms=7)

    plt.tight_layout()
    plt.pause(0.05)
    return fig, line_odo, line_ekf, scat, status_txt, ax_t


def _refresh(fig, line_odo, line_ekf, scat, status_txt, ax_t,
             slam, ctrl, last_t, frames=None):
    now = time.perf_counter()
    if now - last_t[0] < PLOT_INTERVAL_S:
        return
    last_t[0] = now

    if len(slam.odo_trail) > 1:
        oa = np.array(slam.odo_trail)
        line_odo.set_data(oa[:, 0], oa[:, 1])

    if len(slam.ekf_trail) > 1:
        ea = np.array(slam.ekf_trail)
        line_ekf.set_data(ea[:, 0], ea[:, 1])

        # Auto-zoom to include all trails + goal
        all_x = list(ea[:, 0]) + [ctrl.gx, 0]
        all_y = list(ea[:, 1]) + [ctrl.gy, 0]
        pad = 0.3
        ax_t.set_xlim(min(all_x) - pad, max(all_x) + pad)
        ax_t.set_ylim(min(all_y) - pad, max(all_y) + pad)

    if slam.session_log:
        pts = scan_to_xy(slam.session_log[-1])
        if len(pts) > 0:
            scat.set_data(pts[:, 0], pts[:, 1])

    ex, ey, eth = slam.ekf.state
    dist_goal = math.hypot(ctrl.gx - ex, ctrl.gy - ey)
    mode = "GEO" if USE_GEOMETRIC_INIT else "ODO"
    status_txt.set_text(
        f"EKF: ({ex:.2f}, {ey:.2f}, {math.degrees(eth):.0f}°)\n"
        f"Odo: ({slam.odo_x:.2f}, {slam.odo_y:.2f})\n"
        f"Dist to goal: {dist_goal:.2f} m\n"
        f"ICP ok/rej: {slam.icp_accepted}/{slam.icp_rejected}  [{mode}]"
    )
    fig.canvas.draw_idle()
    fig.canvas.flush_events()
    plt.pause(0.001)

    if frames is not None:
        buf = fig.canvas.buffer_rgba()
        w, h = fig.canvas.get_width_height()
        import numpy as _np
        frame = _np.frombuffer(buf, dtype=_np.uint8).reshape(h, w, 4)
        frames.append(frame[:, :, :3].copy())  # drop alpha, keep RGB


# ==============================================================================
#  Post-run comparison plot
# ==============================================================================

def plot_and_save(slam: LiveSLAM, goal_x, goal_y, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    odo_x, odo_y, odo_th = hs.compute_odometry_trajectory(slam.session_log)
    icp_meas, _, _, _    = hs.compute_icp_trajectory(slam.session_log)
    ekf_x, ekf_y, ekf_th = hs.run_ekf_icp(slam.session_log, icp_meas)

    n      = min(len(odo_x), len(ekf_x))
    t_axis = np.arange(n) * 0.1

    odo_goal_dist = math.hypot(odo_x[-1] - goal_x, odo_y[-1] - goal_y)
    ekf_goal_dist = math.hypot(ekf_x[-1] - goal_x, ekf_y[-1] - goal_y)
    odo_ekf_diff  = [math.hypot(odo_x[i] - ekf_x[i], odo_y[i] - ekf_y[i]) for i in range(n)]

    fig = plt.figure(figsize=(15, 9))
    fig.suptitle(
        f"Real Navigation  |  USE_GEOMETRIC_INIT={USE_GEOMETRIC_INIT}  |  "
        f"ICP pairs accepted: {slam.icp_accepted}  rejected: {slam.icp_rejected}\n"
        f"Goal: ({goal_x:.2f}, {goal_y:.2f})   "
        f"Odo final dist to goal: {odo_goal_dist:.3f} m   "
        f"EKF final dist to goal: {ekf_goal_dist:.3f} m",
        fontsize=11
    )
    gs = gridspec.GridSpec(3, 3, figure=fig,
                           left=0.06, right=0.97, top=0.87, bottom=0.07,
                           wspace=0.38, hspace=0.55)

    # ── Panel 1: Main trajectory (2×2) ───────────────────────────────────────
    ax1 = fig.add_subplot(gs[:2, :2])
    ax1.plot(odo_x[:n], odo_y[:n], 'b--', lw=1.6, label='Odometry (predicted)')
    ax1.plot(ekf_x[:n], ekf_y[:n], 'r-',  lw=2.0, label='EKF + ICP (estimated actual)')
    ax1.plot(0, 0, 'ko', ms=10, zorder=7, label='Start')
    ax1.plot(goal_x, goal_y, 'y*', ms=20, zorder=8, label='Goal')
    ax1.plot(odo_x[-1], odo_y[-1], 'b^', ms=9, zorder=6,
             label=f'Odo end ({odo_x[-1]:.2f}, {odo_y[-1]:.2f})')
    ax1.plot(ekf_x[-1], ekf_y[-1], 'r^', ms=9, zorder=6,
             label=f'EKF end ({ekf_x[-1]:.2f}, {ekf_y[-1]:.2f})')
    # Heading arrows
    for xv, yv, tv, col in [(odo_x[-1], odo_y[-1], odo_th[-1], 'blue'),
                              (ekf_x[-1], ekf_y[-1], ekf_th[-1], 'red')]:
        ax1.annotate('',
                     xy=(xv + 0.12*math.cos(tv), yv + 0.12*math.sin(tv)),
                     xytext=(xv, yv),
                     arrowprops=dict(arrowstyle='->', color=col, lw=2.2))
    ax1.set_title('Trajectory: odometry (predicted) vs EKF+ICP (estimated actual)')
    ax1.set_xlabel('X (m)'); ax1.set_ylabel('Y (m)')
    ax1.set_aspect('equal'); ax1.grid(True, ls='--', alpha=0.45)
    ax1.legend(fontsize=9, loc='upper left')

    # ── Panel 2: Odometry vs EKF divergence over time ────────────────────────
    ax2 = fig.add_subplot(gs[2, :2])
    ax2.plot(t_axis, odo_ekf_diff[:n], 'purple', lw=1.8)
    ax2.set_title('Odometry vs EKF position divergence over time')
    ax2.set_xlabel('Time (s)'); ax2.set_ylabel('Difference (m)')
    ax2.grid(True, ls='--', alpha=0.45)

    # ── Panel 3: X over time ─────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(t_axis[:len(odo_x)], odo_x[:n], 'b--', lw=1.4, label='Odo X')
    ax3.plot(t_axis[:len(ekf_x)], ekf_x[:n], 'r-',  lw=1.7, label='EKF X')
    ax3.axhline(goal_x, color='gold', ls=':', lw=1.5, label='Goal X')
    ax3.set_title('X over time'); ax3.set_xlabel('Time (s)'); ax3.set_ylabel('X (m)')
    ax3.legend(fontsize=8); ax3.grid(True, ls='--', alpha=0.45)

    # ── Panel 4: Y over time ─────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.plot(t_axis[:len(odo_y)], odo_y[:n], 'b--', lw=1.4, label='Odo Y')
    ax4.plot(t_axis[:len(ekf_y)], ekf_y[:n], 'r-',  lw=1.7, label='EKF Y')
    ax4.axhline(goal_y, color='gold', ls=':', lw=1.5, label='Goal Y')
    ax4.set_title('Y over time'); ax4.set_xlabel('Time (s)'); ax4.set_ylabel('Y (m)')
    ax4.legend(fontsize=8); ax4.grid(True, ls='--', alpha=0.45)

    # ── Panel 5: Heading over time ────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[2, 2])
    ax5.plot(t_axis[:len(odo_th)], [math.degrees(v) for v in odo_th[:n]],
             'b--', lw=1.4, label='Odo θ')
    ax5.plot(t_axis[:len(ekf_th)], [math.degrees(v) for v in ekf_th[:n]],
             'r-',  lw=1.7, label='EKF θ')
    ax5.set_title('Heading over time'); ax5.set_xlabel('Time (s)'); ax5.set_ylabel('θ (deg)')
    ax5.legend(fontsize=8); ax5.grid(True, ls='--', alpha=0.45)

    ts   = time.strftime("%d_%m_%y_%H_%M_%S")
    path = os.path.join(save_dir, f"real_nav_{ts}.png")
    fig.savefig(path, dpi=150)
    print(f"[nav] Comparison plot saved → {path}")
    plt.show(block=False)
    plt.pause(0.1)


# ==============================================================================
#  Session save (compatible with hybrid_slam.py replay)
# ==============================================================================

def save_session(slam: LiveSLAM, goal_x, goal_y):
    _here    = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(_here, '..', 'data')
    os.makedirs(data_dir, exist_ok=True)
    ts    = time.strftime("%d_%m_%y_%H_%M_%S")
    mode  = "geo" if USE_GEOMETRIC_INIT else "odo"
    fname = os.path.join(data_dir, f"real_nav_{mode}_{ts}.pkl")
    payload = {
        'robot_sensor_signal': slam.session_log,
        'goal_x': goal_x,
        'goal_y': goal_y,
        'use_geometric_init': USE_GEOMETRIC_INIT,
    }
    with open(fname, 'wb') as f:
        pickle.dump(payload, f)
    print(f"[nav] Session saved → {fname}")
    return fname


# ==============================================================================
#  Main drive loop
# ==============================================================================

def run_navigation(goal_x, goal_y, base_speed, timeout_s, estop_m, save_video=False):
    ctrl = ProportionalController(goal_x, goal_y, base_speed=base_speed)
    slam = LiveSLAM()
    sock = make_socket()

    print("=" * 60)
    print("  Real Navigation")
    print(f"  Goal      : ({goal_x:.3f}, {goal_y:.3f}) m from start")
    print(f"  Speed     : {base_speed}")
    print(f"  Timeout   : {timeout_s} s")
    print(f"  E-stop    : {estop_m} m from origin")
    print(f"  ICP init  : {'GEOMETRIC' if USE_GEOMETRIC_INIT else 'ODOMETRY'}")
    print(f"  Arduino   : {ARDUINO_IP}:{PORT}")
    print("  Ctrl-C → e-stop + save")
    print("=" * 60)

    # Wait for first valid packet
    print("[nav] Waiting for Arduino...")
    deadline = time.perf_counter() + 10.0
    while time.perf_counter() < deadline:
        r = recv_sensor(sock)
        if r is not None:
            enc, steer, angles, dists = r
            scan = Scan(enc, steer, angles, dists)
            if scan.num_lidar_rays >= MIN_SCAN_POINTS:
                slam.ingest(scan)
                print(f"[nav] Connected — encoder={enc}  rays={scan.num_lidar_rays}")
                break
    else:
        print("[nav] No packets in 10 s — is the Arduino on?")
        sock.close()
        return slam

    fig, line_odo, line_ekf, scat, status_txt, ax_t = _make_live_fig(goal_x, goal_y)
    last_plot_t = [time.perf_counter()]
    t_start     = time.perf_counter()
    frames      = [] if save_video else None

    goal_reached = False

    _goal_dist_total = math.hypot(goal_x, goal_y)
    _min_travel      = _goal_dist_total * 0.5

    try:
        while True:
            elapsed = time.perf_counter() - t_start

            # ── Control ────────────────────────────────────────────────────────
            ex, ey, eth  = slam.ekf.state
            _odo_travel  = math.hypot(slam.odo_x, slam.odo_y)
            speed, steer = ctrl.compute(ex, ey, eth)
            # Guard: if odometry hasn't covered half the goal distance yet,
            # don't trust a near-goal EKF reading — keep driving at base speed.
            if _odo_travel < _min_travel and speed == 0:
                speed = base_speed
            send_command(sock, speed, steer)

            # ── Receive ────────────────────────────────────────────────────────
            r = recv_sensor(sock)
            if r is not None:
                enc, steer_fb, angles, dists = r
                scan = Scan(enc, steer_fb, angles, dists)
                if scan.num_lidar_rays >= MIN_SCAN_POINTS:
                    slam.ingest(scan)

            # ── Goal check ─────────────────────────────────────────────────────
            if _odo_travel >= _min_travel and ctrl.at_goal(ex, ey):
                print(f"\n[nav] Goal reached!  EKF: ({ex:.3f}, {ey:.3f})  "
                      f"time: {elapsed:.1f} s")
                goal_reached = True
                break

            # ── Timeout / e-stop ───────────────────────────────────────────────
            if elapsed >= timeout_s:
                print(f"\n[nav] Timeout ({timeout_s} s) reached.")
                break
            odo_dist = math.hypot(slam.odo_x, slam.odo_y)
            if odo_dist > estop_m:
                print(f"\n[nav] E-STOP: odometry distance {odo_dist:.2f} m > {estop_m} m")
                break

            _refresh(fig, line_odo, line_ekf, scat, status_txt, ax_t,
                     slam, ctrl, last_plot_t, frames)

    except KeyboardInterrupt:
        print("\n[nav] Ctrl-C — stopping.")

    except Exception as e:
        import traceback
        print(f"\n[nav] EXCEPTION — {type(e).__name__}: {e}")
        traceback.print_exc()

    finally:
        for _ in range(10):
            send_stop(sock)
            time.perf_counter()   # yield
        import time as _t; _t.sleep(0.2)
        sock.close()

    print(f"\n[nav] Summary:")
    print(f"  Scans       : {slam.scan_count}")
    print(f"  ICP ok/rej  : {slam.icp_accepted}/{slam.icp_rejected}")
    ex, ey, eth = slam.ekf.state
    print(f"  Final EKF   : ({ex:.3f}, {ey:.3f}, {math.degrees(eth):.1f}°)")
    print(f"  Final Odo   : ({slam.odo_x:.3f}, {slam.odo_y:.3f})")
    print(f"  EKF→goal    : {math.hypot(goal_x-ex, goal_y-ey):.3f} m")
    print(f"  Odo→goal    : {math.hypot(goal_x-slam.odo_x, goal_y-slam.odo_y):.3f} m")

    if save_video and frames:
        import cv2 as _cv2
        _here    = os.path.dirname(os.path.abspath(__file__))
        vid_dir  = os.path.join(_here, '..', 'results', 'nav_real')
        os.makedirs(vid_dir, exist_ok=True)
        ts       = time.strftime("%d_%m_%y_%H_%M_%S")
        vid_path = os.path.join(vid_dir, f"real_nav_{ts}.mp4")
        h, w     = frames[0].shape[:2]
        writer   = _cv2.VideoWriter(vid_path,
                                    _cv2.VideoWriter_fourcc(*'mp4v'),
                                    5, (w, h))
        for f in frames:
            writer.write(_cv2.cvtColor(f, _cv2.COLOR_RGB2BGR))
        writer.release()
        print(f"[nav] Live video saved → {vid_path}")

    return slam


# ==============================================================================
#  Entry point
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Drive the robot from its current position to a goal "
                    "using SLAM feedback.")
    parser.add_argument('--goal', nargs=2, type=float, required=True,
                        metavar=('X', 'Y'),
                        help='Goal position in metres relative to robot start')
    parser.add_argument('--speed', type=int, default=DEFAULT_BASE_SPEED,
                        help=f'Base drive speed 0-100 (default {DEFAULT_BASE_SPEED})')
    parser.add_argument('--timeout', type=float, default=DEFAULT_TIMEOUT_S,
                        help=f'Hard timeout in seconds (default {DEFAULT_TIMEOUT_S})')
    parser.add_argument('--estop-m', type=float, default=DEFAULT_ESTOP_M,
                        dest='estop_m',
                        help=f'E-stop distance from origin in metres '
                             f'(default {DEFAULT_ESTOP_M})')
    parser.add_argument('--save-video', action='store_true',
                        help='Save live tracking window as an mp4 video')
    args = parser.parse_args()

    gx, gy = args.goal
    slam   = run_navigation(gx, gy, args.speed, args.timeout, args.estop_m,
                            save_video=args.save_video)

    if slam.scan_count < 5:
        print("[nav] Too few scans — skipping plots.")
        return

    _here    = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(_here, '..', 'results', 'nav_real')
    save_session(slam, gx, gy)
    plot_and_save(slam, gx, gy, save_dir)
    plt.show()


if __name__ == '__main__':
    main()
