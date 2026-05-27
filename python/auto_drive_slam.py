# ==============================================================
#  auto_drive_slam.py
#  Autonomous drive sequence + live trajectory comparison.
#
#  The script drives the robot through a pre-defined sequence of
#  (speed, steering, duration) segments, runs the Hybrid SLAM
#  pipeline in real-time, and at the end generates a side-by-side
#  comparison of:
#    - Odometry trajectory
#    - EKF + ICP trajectory
#    - Encoder-derived distance vs time
#
#  The session is saved to ../data/ in the same format as the
#  existing test pkl files so hybrid_slam.py can replay it.
#
#  Usage:
#    python auto_drive_slam.py
#    python auto_drive_slam.py --sequence right_circle
#    python auto_drive_slam.py --sequence straight
#    python auto_drive_slam.py --sequence left_circle
#    python auto_drive_slam.py --sequence figure_eight   (two loops)
#
#  Safety:
#    - Press Ctrl-C at any time to e-stop (speed → 0) and save.
#    - The robot automatically stops at the end of the sequence.
#    - ESTOP_DISTANCE_M: if odometry distance exceeds this, stop.
# ==============================================================

import argparse
import math
import os
import pickle
import socket
import time
from collections import deque

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

import hybrid_slam as hs
from hybrid_slam import (
    wrap_angle, encoder_counts_to_distance,
    scan_to_xy, rotate_points, icp,
    extract_lines, geometric_align,
    WHEEL_BASE, MIN_SCAN_POINTS,
    ICP_ERROR_THRESHOLD, MIN_DS, CONSISTENCY_THRESH,
    USE_GEOMETRIC_INIT,
)
from live_slam import (
    make_socket, recv_sensor, send_stop, Scan,
    LiveSLAM, LIVE_ICP_STRIDE,
    LOCAL_IP, ARDUINO_IP, PORT, SEND_EVERY,
)

# ==============================================================================
#  Drive sequences
#  Each segment: (speed 0-100, steering_deg, duration_seconds)
#  speed=0 means stop-and-coast for that duration (useful for pauses).
#  Positive steering = right, negative = left (matches Arduino convention).
# ==============================================================================

SEQUENCES = {
    'straight': [
        (62, 0, 8.0),
        ( 0, 0, 1.0),
    ],
    'right_circle': [
        (62, 6, 10.0),
        ( 0, 0, 1.0),
    ],
    'left_circle': [
        (57, -12, 10.0),
        (  0,  0,  1.0),
    ],
    'right_hard': [
        (72, 20, 10.0),
        ( 0,  0,  1.0),
    ],
    'left_hard': [
        (72, -20, 10.0),
        ( 0,   0,  1.0),
    ],
    'figure_eight': [
        (62,  12, 9.0),   # first loop — right
        (62, -12, 9.0),   # second loop — left
        ( 0,   0, 1.0),
    ],
    'square': [
        (62,  0, 3.0),    # forward
        ( 0,  0, 0.5),
        (62, 20, 2.5),    # turn right ~90°
        ( 0,  0, 0.5),
        (62,  0, 3.0),
        ( 0,  0, 0.5),
        (62, 20, 2.5),
        ( 0,  0, 0.5),
        (62,  0, 3.0),
        ( 0,  0, 0.5),
        (62, 20, 2.5),
        ( 0,  0, 0.5),
        (62,  0, 3.0),
        ( 0,  0, 1.0),
    ],
}

DEFAULT_SEQUENCE = 'right_circle'

# Safety cutoff — stop if odometry distance from start exceeds this
ESTOP_DISTANCE_M = 8.0

# ==============================================================================
#  Control sender
# ==============================================================================

_last_send_time = 0.0

def send_command(sock, speed: int, steering: int):
    global _last_send_time
    now = time.perf_counter()
    if now - _last_send_time >= SEND_EVERY:
        msg = f"{int(speed)}, {int(steering)}\n"
        sock.sendto(msg.encode(), (ARDUINO_IP, PORT))
        _last_send_time = now


# ==============================================================================
#  Post-run comparison plots
# ==============================================================================

def plot_comparison(slam: LiveSLAM, sequence_name: str, save_path: str):
    """
    Generate a 6-panel figure comparing odometry vs EKF+ICP.
    Saves to save_path and displays interactively.
    """
    odo_arr = np.array(list(slam.odo_trail))
    ekf_arr = np.array(list(slam.ekf_trail))
    n       = len(slam.session_log)
    t_axis  = np.arange(n) * 0.1     # seconds (100 ms per scan)

    odo_x_t = []; odo_y_t = []; odo_th_t = []
    ekf_x_t = []; ekf_y_t = []; ekf_th_t = []
    dist_t  = []

    # Re-run odometry + EKF over full log for per-timestep plots
    ox, oy, oth = hs.compute_odometry_trajectory(slam.session_log)
    icp_meas, _, _, _ = hs.compute_icp_trajectory(slam.session_log)
    ex, ey, eth = hs.run_ekf_icp(slam.session_log, icp_meas)

    # Cumulative odometry distance
    cum_dist = [0.0]
    for i in range(1, len(slam.session_log)):
        dc = slam.session_log[i].encoder_counts - slam.session_log[i-1].encoder_counts
        cum_dist.append(cum_dist[-1] + abs(encoder_counts_to_distance(dc)))

    timesteps = list(range(len(slam.session_log)))

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(
        f"Auto-Drive SLAM Comparison — sequence: '{sequence_name}'\n"
        f"USE_GEOMETRIC_INIT={USE_GEOMETRIC_INIT}   "
        f"ICP pairs accepted: {slam.icp_accepted}   "
        f"rejected: {slam.icp_rejected}   "
        f"scans: {slam.scan_count}",
        fontsize=12
    )
    gs = gridspec.GridSpec(3, 3, figure=fig,
                           left=0.07, right=0.97, top=0.88, bottom=0.07,
                           wspace=0.38, hspace=0.55)

    # ── 1. Main trajectory ────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[:2, :2])
    ax1.plot(ox, oy, 'b--', lw=1.8, label='Odometry')
    ax1.plot(ex, ey, 'g-',  lw=2.2, label='EKF + ICP')
    ax1.plot(ox[0], oy[0], 'ko', ms=10, zorder=6, label='Start')
    ax1.plot(ox[-1], oy[-1], 'b^', ms=9, zorder=6,
             label=f'End odo ({ox[-1]:.2f}, {oy[-1]:.2f}) m')
    ax1.plot(ex[-1], ey[-1], 'g^', ms=9, zorder=6,
             label=f'End EKF ({ex[-1]:.2f}, {ey[-1]:.2f}) m')
    # Heading arrows at end
    for xv, yv, tv, col in [(ox[-1], oy[-1], oth[-1], 'blue'),
                              (ex[-1], ey[-1], eth[-1], 'green')]:
        ax1.annotate('', xy=(xv + 0.15*math.cos(tv), yv + 0.15*math.sin(tv)),
                     xytext=(xv, yv),
                     arrowprops=dict(arrowstyle='->', color=col, lw=2.5))
    ax1.set_title('Trajectory (top-down view)')
    ax1.set_xlabel('X (m)'); ax1.set_ylabel('Y (m)')
    ax1.set_aspect('equal'); ax1.grid(True, ls='--', alpha=0.5)
    ax1.legend(fontsize=8, loc='upper left')

    # ── 2. Cumulative distance ────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[2, :2])
    ax2.plot(t_axis[:len(cum_dist)], cum_dist, 'purple', lw=1.8)
    ax2.set_title('Cumulative distance driven (odometry)')
    ax2.set_xlabel('Time (s)'); ax2.set_ylabel('Distance (m)')
    ax2.grid(True, ls='--', alpha=0.5)

    # ── 3. X over time ───────────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(t_axis[:len(ox)], ox, 'b--', lw=1.5, label='Odo X')
    ax3.plot(t_axis[:len(ex)], ex, 'g-',  lw=1.8, label='EKF X')
    ax3.set_title('X over time'); ax3.set_xlabel('Time (s)'); ax3.set_ylabel('X (m)')
    ax3.legend(fontsize=8); ax3.grid(True, ls='--', alpha=0.5)

    # ── 4. Y over time ───────────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.plot(t_axis[:len(oy)], oy, 'b--', lw=1.5, label='Odo Y')
    ax4.plot(t_axis[:len(ey)], ey, 'g-',  lw=1.8, label='EKF Y')
    ax4.set_title('Y over time'); ax4.set_xlabel('Time (s)'); ax4.set_ylabel('Y (m)')
    ax4.legend(fontsize=8); ax4.grid(True, ls='--', alpha=0.5)

    # ── 5. Heading over time ─────────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[2, 2])
    ax5.plot(t_axis[:len(oth)], [math.degrees(v) for v in oth], 'b--', lw=1.5, label='Odo θ')
    ax5.plot(t_axis[:len(eth)], [math.degrees(v) for v in eth], 'g-',  lw=1.8, label='EKF θ')
    ax5.set_title('Heading over time'); ax5.set_xlabel('Time (s)'); ax5.set_ylabel('θ (deg)')
    ax5.legend(fontsize=8); ax5.grid(True, ls='--', alpha=0.5)

    fig.savefig(save_path, dpi=150)
    print(f"[auto_drive] Comparison plot saved → {save_path}")

    plt.show(block=False)
    plt.pause(0.1)


def plot_map(slam: LiveSLAM, sequence_name: str, save_path: str):
    """Build and save a geometric map using the EKF poses."""
    icp_meas, _, _, _ = hs.compute_icp_trajectory(slam.session_log)
    ex, ey, eth = hs.run_ekf_icp(slam.session_log, icp_meas)
    ekf_poses = list(zip(ex, ey, eth))
    gpts = hs.build_global_map(slam.session_log, ekf_poses)

    fig, ax = plt.subplots(figsize=(9, 8))
    if len(gpts) > 0:
        ax.scatter(gpts[:, 0], gpts[:, 1], s=1, c='gray', alpha=0.25,
                   label=f'LiDAR map ({len(gpts)} pts)')
    odo_arr = np.array(list(slam.odo_trail))
    ax.plot(odo_arr[:, 0], odo_arr[:, 1], 'b--', lw=1.2, alpha=0.7, label='Odometry')
    ax.plot(ex, ey, 'g-', lw=2.0, label='EKF trajectory')
    ax.plot(ex[0], ey[0], 'ko', ms=9, label='Start')
    ax.set_title(f"Geometric Map — '{sequence_name}'  (USE_GEOMETRIC_INIT={USE_GEOMETRIC_INIT})")
    ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)')
    ax.set_aspect('equal'); ax.grid(True, ls='--', alpha=0.5)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"[auto_drive] Map plot saved → {save_path}")
    plt.show(block=False)
    plt.pause(0.1)


# ==============================================================================
#  Session save
# ==============================================================================

def save_session(slam: LiveSLAM, sequence_name: str) -> str:
    _here    = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(_here, '..', 'data')
    os.makedirs(data_dir, exist_ok=True)
    ts    = time.strftime("%d_%m_%y_%H_%M_%S")
    mode  = "geo" if USE_GEOMETRIC_INIT else "odo"
    fname = os.path.join(data_dir,
                         f"auto_{sequence_name}_{mode}_{ts}.pkl")
    payload = {
        'robot_sensor_signal': slam.session_log,
        'odo_trail':           list(slam.odo_trail),
        'ekf_trail':           list(slam.ekf_trail),
        'icp_accepted':        slam.icp_accepted,
        'icp_rejected':        slam.icp_rejected,
        'use_geometric_init':  USE_GEOMETRIC_INIT,
        'sequence_name':       sequence_name,
    }
    with open(fname, 'wb') as f:
        pickle.dump(payload, f)
    print(f"[auto_drive] Session saved → {fname}")
    return fname


# ==============================================================================
#  Main
# ==============================================================================

def main(sequence_name: str):
    sequence = SEQUENCES[sequence_name]
    total_drive_time = sum(d for _, _, d in sequence)

    print("=" * 60)
    print("  Auto-Drive SLAM")
    print(f"  Sequence : '{sequence_name}'  ({total_drive_time:.1f} s total)")
    print(f"  ICP init : {'GEOMETRIC' if USE_GEOMETRIC_INIT else 'ODOMETRY'}")
    print(f"  E-stop   : {ESTOP_DISTANCE_M} m from start")
    print(f"  Arduino  : {ARDUINO_IP}:{PORT}")
    print("  Ctrl-C   → immediate e-stop + save")
    print("=" * 60)

    sock = make_socket()
    slam = LiveSLAM()

    # ── Wait for first valid packet ───────────────────────────────────────────
    print("[auto_drive] Waiting for Arduino packets...")
    deadline = time.perf_counter() + 10.0
    while time.perf_counter() < deadline:
        r = recv_sensor(sock)
        if r is not None:
            encoder, steering, angles, distances = r
            first_scan = Scan(encoder, steering, angles, distances)
            if first_scan.num_lidar_rays >= MIN_SCAN_POINTS:
                slam.ingest(first_scan)
                print(f"[auto_drive] Connected. Encoder={encoder}  rays={first_scan.num_lidar_rays}")
                break
    else:
        print("[auto_drive] No packets received in 10 s — is the Arduino on?")
        sock.close()
        return

    # ── Live plot (non-blocking) ──────────────────────────────────────────────
    fig = plt.figure(figsize=(10, 7))
    fig.canvas.manager.set_window_title("Auto-Drive SLAM — live")
    ax_traj = fig.add_subplot(1, 2, 1)
    ax_scan = fig.add_subplot(1, 2, 2)

    ax_traj.set_title("Live Trajectory"); ax_traj.set_xlabel("X (m)"); ax_traj.set_ylabel("Y (m)")
    ax_traj.set_aspect('equal'); ax_traj.grid(True, ls='--', alpha=0.4)
    line_odo, = ax_traj.plot([], [], 'b--', lw=1.5, label='Odometry')
    line_ekf, = ax_traj.plot([], [], 'g-',  lw=2.0, label='EKF + ICP')
    ax_traj.plot(0, 0, 'ko', ms=8, label='Start')
    ax_traj.legend(fontsize=9)
    status_txt = ax_traj.text(0.02, 0.97, '', transform=ax_traj.transAxes,
                               va='top', ha='left', fontsize=8,
                               bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.8))

    ax_scan.set_title("Current LiDAR"); ax_scan.set_aspect('equal')
    ax_scan.set_xlim(-4, 4); ax_scan.set_ylim(-4, 4)
    ax_scan.grid(True, ls='--', alpha=0.4)
    scat_scan = ax_scan.scatter([], [], s=4, c='steelblue', alpha=0.7)
    ax_scan.plot(0, 0, 'ro', ms=7)

    plt.tight_layout()
    plt.pause(0.05)

    last_plot = time.perf_counter()
    PLOT_INTERVAL = 0.20

    estop_triggered = False
    saved_path = None

    def refresh_plot(seg_idx, seg_elapsed, seg_total, cmd_spd, cmd_steer):
        nonlocal last_plot
        now = time.perf_counter()
        if now - last_plot < PLOT_INTERVAL:
            return
        last_plot = now

        if len(slam.odo_trail) > 1:
            oa = np.array(slam.odo_trail)
            line_odo.set_data(oa[:, 0], oa[:, 1])
            ax_traj.set_xlim(oa[:, 0].min()-1, oa[:, 0].max()+1)
            ax_traj.set_ylim(oa[:, 1].min()-1, oa[:, 1].max()+1)
        if len(slam.ekf_trail) > 1:
            ea = np.array(slam.ekf_trail)
            line_ekf.set_data(ea[:, 0], ea[:, 1])

        if slam.session_log:
            pts = scan_to_xy(slam.session_log[-1])
            if len(pts) > 0:
                scat_scan.set_offsets(pts)

        mode = "GEO" if USE_GEOMETRIC_INIT else "ODO"
        status_txt.set_text(
            f"Seg {seg_idx+1}/{len(sequence)}  spd={cmd_spd} steer={cmd_steer}°\n"
            f"Elapsed {seg_elapsed:.1f}/{seg_total:.1f} s\n"
            f"ICP ok/rej: {slam.icp_accepted}/{slam.icp_rejected}  [{mode}]\n"
            f"EKF ({slam.ekf.state[0]:.2f},{slam.ekf.state[1]:.2f},{math.degrees(slam.ekf.state[2]):.0f}°)\n"
            f"Odo ({slam.odo_x:.2f},{slam.odo_y:.2f},{math.degrees(slam.odo_theta):.0f}°)"
        )
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        plt.pause(0.001)

    # ── Drive sequence ────────────────────────────────────────────────────────
    try:
        for seg_idx, (cmd_speed, cmd_steering, duration) in enumerate(sequence):
            print(f"\n[auto_drive] Segment {seg_idx+1}/{len(sequence)}: "
                  f"speed={cmd_speed}  steering={cmd_steering}°  duration={duration:.1f}s")
            seg_start = time.perf_counter()

            while True:
                elapsed = time.perf_counter() - seg_start
                if elapsed >= duration:
                    break

                # Send drive command
                send_command(sock, cmd_speed, cmd_steering)

                # Receive sensor data
                result = recv_sensor(sock)
                if result is not None:
                    encoder, steering_fb, angles, distances = result
                    scan = Scan(encoder, steering_fb, angles, distances)
                    if scan.num_lidar_rays >= MIN_SCAN_POINTS:
                        slam.ingest(scan)

                # E-stop if robot has gone too far
                dist_from_start = math.hypot(slam.odo_x, slam.odo_y)
                if dist_from_start > ESTOP_DISTANCE_M:
                    print(f"\n[auto_drive] E-STOP: odometry distance {dist_from_start:.2f} m "
                          f"> limit {ESTOP_DISTANCE_M} m")
                    estop_triggered = True
                    break

                refresh_plot(seg_idx, elapsed, duration, cmd_speed, cmd_steering)

            if estop_triggered:
                break

        print("\n[auto_drive] Sequence complete. Stopping robot.")

    except KeyboardInterrupt:
        print("\n[auto_drive] Ctrl-C — stopping robot.")

    finally:
        # Always stop the robot first
        for _ in range(10):
            send_stop(sock)
            time.sleep(0.02)
        sock.close()

        print(f"\n[auto_drive] Summary:")
        print(f"  Scans collected : {slam.scan_count}")
        print(f"  ICP accepted    : {slam.icp_accepted}")
        print(f"  ICP rejected    : {slam.icp_rejected}")
        print(f"  Final Odo pose  : ({slam.odo_x:.3f}, {slam.odo_y:.3f}, "
              f"{math.degrees(slam.odo_theta):.1f}°)")
        print(f"  Final EKF pose  : ({slam.ekf.state[0]:.3f}, {slam.ekf.state[1]:.3f}, "
              f"{math.degrees(slam.ekf.state[2]):.1f}°)")

        if slam.scan_count < 5:
            print("[auto_drive] Too few scans — skipping plots.")
            return

        saved_path = save_session(slam, sequence_name)

        _here     = os.path.dirname(os.path.abspath(__file__))
        _res_dir  = os.path.join(_here, '..', 'results', 'auto_drive')
        os.makedirs(_res_dir, exist_ok=True)
        ts = time.strftime("%d_%m_%y_%H_%M_%S")

        print("\n[auto_drive] Generating post-run plots...")
        plot_comparison(slam, sequence_name,
                        os.path.join(_res_dir, f"{sequence_name}_comparison_{ts}.png"))
        plot_map(slam, sequence_name,
                 os.path.join(_res_dir, f"{sequence_name}_map_{ts}.png"))

        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Drive the robot through a sequence and compare SLAM estimates.")
    parser.add_argument(
        '--sequence', '-s',
        choices=list(SEQUENCES.keys()),
        default=DEFAULT_SEQUENCE,
        help=f"Drive sequence to run (default: {DEFAULT_SEQUENCE})"
    )
    args = parser.parse_args()
    main(args.sequence)
