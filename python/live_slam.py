# ==============================================================
#  live_slam.py
#  Real-time Hybrid SLAM visualiser — runs on hardware.
#
#  Connects to the Arduino via UDP, receives LiDAR + encoder
#  data at ~10 Hz, and displays a live updating plot showing:
#    - Odometry trajectory (blue dashed)
#    - EKF + ICP trajectory (green solid)
#    - Current LiDAR scan in sensor frame (right panel)
#    - Live ICP pair error bar and accepted/rejected count
#
#  Controls (keyboard in the matplotlib window):
#    SPACE  — pause / resume display updates (data still collected)
#    q / Q  — quit and save the session recording to ../data/
#    r / R  — reset trajectory to origin (keeps connection)
#
#  Usage:
#    python live_slam.py
#
#  Hardware required:
#    Arduino Giga R1 at 192.168.0.200:4010 (UDP)
#    RPLiDAR providing 50 rays per packet
# ==============================================================

import math
import os
import pickle
import socket
import time
from collections import deque

import matplotlib
matplotlib.use('TkAgg')          # works on macOS; fall back to Qt5Agg if needed
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# ── Import shared SLAM functions from hybrid_slam.py ──────────────────────────
import hybrid_slam as hs
from hybrid_slam import (
    wrap_angle, encoder_counts_to_distance,
    scan_to_xy, rotate_points, icp,
    extract_lines, geometric_align,
    WHEEL_BASE, ENCODER_SCALE, MIN_SCAN_POINTS,
    ICP_ERROR_THRESHOLD, MIN_DS, CONSISTENCY_THRESH,
    USE_GEOMETRIC_INIT, STEER_SCALE,
)

# ── Network settings (match parameters.py) ────────────────────────────────────
ARDUINO_IP   = "192.168.0.197"
LOCAL_IP     = "192.168.0.200"
PORT         = 4010
BUFFER_SIZE  = 8192
UDP_TIMEOUT  = 0.012          # seconds — non-blocking receive
SEND_EVERY   = 0.10           # seconds between control commands to Arduino

# ── ICP stride in live mode ───────────────────────────────────────────────────
# Accumulate this many NEW scans before running an ICP pair.
# Lower = more frequent corrections but heavier CPU load.
LIVE_ICP_STRIDE = 4

# ── Display ───────────────────────────────────────────────────────────────────
TRAIL_LEN    = 500            # max trajectory points kept for plotting
PLOT_RADIUS  = 6.0            # metres — axes range around current pose
SCAN_RANGE   = 4.0            # metres — LiDAR scan plot axis range

# ==============================================================================
#  UDP helpers (standalone — no dependency on robot_python_code.py)
# ==============================================================================

def make_socket():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((LOCAL_IP, PORT))
    sock.settimeout(UDP_TIMEOUT)
    return sock


def recv_sensor(sock):
    """
    Read one UDP packet from the Arduino and parse it.
    Returns (encoder_counts, steering_deg, angles_list, distances_list)
    or None on timeout / parse error.
    """
    try:
        data, _ = sock.recvfrom(BUFFER_SIZE)
        parts = data.decode().split(',')
        if len(parts) < 3:
            return None
        encoder  = int(float(parts[0]))
        steering = int(float(parts[1]))
        n_rays   = int(float(parts[2]))
        if len(parts) < 3 + n_rays * 2:
            return None
        angles    = []
        distances = []
        for i in range(n_rays):
            idx = 3 + i * 2
            angles.append(float(parts[idx]))
            distances.append(float(parts[idx + 1]))
        return encoder, steering, angles, distances
    except (socket.timeout, ValueError, IndexError):
        return None


def send_stop(sock):
    """Send speed=0, steering=0 (safe stop)."""
    msg = "0, 0\n"
    sock.sendto(msg.encode(), (ARDUINO_IP, PORT))


# ==============================================================================
#  Minimal RobotSensorSignal wrapper so scan_to_xy() works unchanged
# ==============================================================================

class Scan:
    def __init__(self, encoder, steering, angles, distances):
        self.encoder_counts = encoder
        self.steering       = steering
        self.num_lidar_rays = len(angles)
        self.angles         = angles
        self.distances      = distances


# ==============================================================================
#  Live SLAM state
# ==============================================================================

class LiveSLAM:
    def __init__(self):
        # Odometry state
        self.odo_x     = 0.0
        self.odo_y     = 0.0
        self.odo_theta = 0.0

        # EKF state
        self.ekf = hs.EKF(x0=[0.0, 0.0, 0.0], sigma0=np.eye(3) * 1e-4)

        # Scan buffer for ICP (keep last LIVE_ICP_STRIDE + 1 scans)
        self.scan_buffer    = deque(maxlen=LIVE_ICP_STRIDE + 2)
        self.encoder_buffer = deque(maxlen=LIVE_ICP_STRIDE + 2)
        self.steer_buffer   = deque(maxlen=LIVE_ICP_STRIDE + 2)
        self.scan_count     = 0   # total valid scans received

        # Trajectory trails (for plotting)
        self.odo_trail = deque(maxlen=TRAIL_LEN)
        self.ekf_trail = deque(maxlen=TRAIL_LEN)
        self.odo_trail.append((0.0, 0.0))
        self.ekf_trail.append((0.0, 0.0))

        # ICP stats
        self.icp_accepted  = 0
        self.icp_rejected  = 0
        self.icp_errors    = deque(maxlen=30)   # recent errors for bar chart

        # Raw session log (for offline replay / saving)
        self.session_log   = []                 # list of Scan objects
        self.last_encoder  = None

        # Flags
        self.paused        = False

    # ── Odometry step (called every scan) ──────────────────────────────────
    def _odo_step(self, delta_counts, steering_deg):
        ds     = encoder_counts_to_distance(delta_counts)
        phi    = math.radians(steering_deg * STEER_SCALE)
        dtheta = (ds / WHEEL_BASE) * math.tan(phi)
        self.odo_theta = wrap_angle(self.odo_theta + dtheta)
        self.odo_x   += ds * math.cos(self.odo_theta)
        self.odo_y   += ds * math.sin(self.odo_theta)
        return ds, dtheta

    # ── Process one incoming scan ───────────────────────────────────────────
    def ingest(self, scan: Scan):
        if self.last_encoder is None:
            self.last_encoder = scan.encoder_counts
            self.scan_buffer.append(scan)
            self.encoder_buffer.append(scan.encoder_counts)
            self.steer_buffer.append(scan.steering)
            self.session_log.append(scan)
            return

        delta_enc = scan.encoder_counts - self.last_encoder
        self.last_encoder = scan.encoder_counts

        # ── Odometry update ───────────────────────────────────────────────
        ds, _ = self._odo_step(delta_enc, scan.steering)

        # ── EKF prediction (every step) ───────────────────────────────────
        self.ekf.predict(delta_enc, scan.steering)

        # ── Buffer this scan ──────────────────────────────────────────────
        self.scan_buffer.append(scan)
        self.encoder_buffer.append(scan.encoder_counts)
        self.steer_buffer.append(scan.steering)
        self.scan_count += 1
        self.session_log.append(scan)

        # ── Attempt ICP every LIVE_ICP_STRIDE scans ───────────────────────
        if self.scan_count % LIVE_ICP_STRIDE == 0 and len(self.scan_buffer) >= LIVE_ICP_STRIDE + 1:
            self._try_icp()

        # ── Record trails ─────────────────────────────────────────────────
        self.odo_trail.append((self.odo_x, self.odo_y))
        self.ekf_trail.append((self.ekf.state[0], self.ekf.state[1]))

    # ── ICP between oldest and newest buffered scan ─────────────────────────
    def _try_icp(self):
        scan_i = self.scan_buffer[0]
        scan_j = self.scan_buffer[-1]

        pts_i = scan_to_xy(scan_i)
        pts_j = scan_to_xy(scan_j)

        if len(pts_i) < MIN_SCAN_POINTS or len(pts_j) < MIN_SCAN_POINTS:
            self.icp_rejected += 1
            return

        # Odometry between the two scans
        enc_diff    = scan_j.encoder_counts - scan_i.encoder_counts
        ds          = encoder_counts_to_distance(enc_diff)
        dtheta_odom = (ds / WHEEL_BASE) * math.tan(math.radians(scan_i.steering * STEER_SCALE))

        if abs(ds) < MIN_DS:
            return  # robot barely moved — no information

        # ICP initialisation
        if USE_GEOMETRIC_INIT:
            li = extract_lines(pts_i)
            lj = extract_lines(pts_j)
            dth_init, dx_init, dy_init, ok = geometric_align(li, lj)
            if not ok:
                dth_init, dx_init, dy_init = -dtheta_odom, -ds, 0.0
        else:
            dth_init, dx_init, dy_init = -dtheta_odom, -ds, 0.0

        pts_i_init = rotate_points(pts_i, dth_init) + np.array([dx_init, dy_init])
        _, R_icp, t_icp, err = icp(pts_i_init, pts_j)

        if err > ICP_ERROR_THRESHOLD:
            self.icp_rejected += 1
            return

        # Compose total scan-to-scan transform
        R_init = np.array([[math.cos(dth_init), -math.sin(dth_init)],
                            [math.sin(dth_init),  math.cos(dth_init)]])
        R_total    = R_icp @ R_init
        t_total    = R_icp @ np.array([dx_init, dy_init]) + t_icp
        dth_phys   = -math.atan2(R_total[1, 0], R_total[0, 0])
        t_global_d = -t_total  # physical robot displacement in sensor-j frame

        # Consistency check
        if abs(wrap_angle(dth_phys - dtheta_odom)) > CONSISTENCY_THRESH:
            self.icp_rejected += 1
            return

        # Build absolute ICP pose from EKF current state + ICP delta
        theta_icp = wrap_angle(self.ekf.state[2] + dth_phys)
        R_g = np.array([[math.cos(theta_icp), -math.sin(theta_icp)],
                          [math.sin(theta_icp),  math.cos(theta_icp)]])
        dp  = R_g @ t_global_d
        x_icp = self.ekf.state[0] + dp[0]
        y_icp = self.ekf.state[1] + dp[1]

        # EKF correction
        self.ekf.correct([x_icp, y_icp, theta_icp], err)

        self.icp_accepted += 1
        self.icp_errors.append(err)


# ==============================================================================
#  Live plot
# ==============================================================================

class LivePlot:
    def __init__(self, slam: LiveSLAM):
        self.slam = slam

        self.fig = plt.figure(figsize=(14, 7))
        self.fig.canvas.manager.set_window_title("Live SLAM — hybrid_slam.py")
        gs = gridspec.GridSpec(2, 3, figure=self.fig,
                               left=0.06, right=0.97, top=0.93, bottom=0.08,
                               wspace=0.35, hspace=0.45)

        # Top-left: main trajectory
        self.ax_traj = self.fig.add_subplot(gs[:, :2])
        # Top-right: current LiDAR scan
        self.ax_scan = self.fig.add_subplot(gs[0, 2])
        # Bottom-right: ICP error history
        self.ax_err  = self.fig.add_subplot(gs[1, 2])

        self._init_trajectory_axes()
        self._init_scan_axes()
        self._init_error_axes()

        # Key bindings
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)
        self.running = True

    def _init_trajectory_axes(self):
        ax = self.ax_traj
        ax.set_title("Live Trajectory", fontsize=12)
        ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
        ax.set_aspect('equal')
        ax.grid(True, ls='--', alpha=0.4)
        self.line_odo,  = ax.plot([], [], 'b--', lw=1.5, label='Odometry')
        self.line_ekf,  = ax.plot([], [], 'g-',  lw=2.0, label='EKF + ICP')
        self.pt_start,  = ax.plot([0], [0], 'ko', ms=9, zorder=6, label='Start')
        self.pt_ekf,    = ax.plot([], [], 'g^',  ms=9,  zorder=6)
        self.pt_odo,    = ax.plot([], [], 'b^',  ms=9,  zorder=6)
        self.arrow_ekf  = None
        ax.legend(fontsize=9, loc='upper left')
        self.status_txt = ax.text(
            0.02, 0.97, '', transform=ax.transAxes,
            va='top', ha='left', fontsize=9,
            bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))

    def _init_scan_axes(self):
        ax = self.ax_scan
        ax.set_title("Current LiDAR scan", fontsize=10)
        ax.set_xlim(-SCAN_RANGE, SCAN_RANGE)
        ax.set_ylim(-SCAN_RANGE, SCAN_RANGE)
        ax.set_aspect('equal')
        ax.grid(True, ls='--', alpha=0.4)
        self.scat_scan = ax.scatter([], [], s=4, c='steelblue', alpha=0.7)
        ax.plot(0, 0, 'ro', ms=6)           # robot origin

    def _init_error_axes(self):
        ax = self.ax_err
        ax.set_title("Recent ICP errors", fontsize=10)
        ax.set_xlabel("Pair #"); ax.set_ylabel("Mean error (m)")
        ax.axhline(ICP_ERROR_THRESHOLD, color='red', ls='--', lw=1.2,
                   label=f'Threshold ({ICP_ERROR_THRESHOLD} m)')
        ax.legend(fontsize=8)
        ax.grid(True, ls='--', alpha=0.4)
        self.bar_errs = None

    # ── Refresh all axes with current SLAM state ───────────────────────────
    def refresh(self):
        slam = self.slam
        if slam.paused:
            return

        # ── Trajectory ────────────────────────────────────────────────────
        if len(slam.odo_trail) > 1:
            odo_xy = np.array(slam.odo_trail)
            self.line_odo.set_data(odo_xy[:, 0], odo_xy[:, 1])
            self.pt_odo.set_data([odo_xy[-1, 0]], [odo_xy[-1, 1]])

        if len(slam.ekf_trail) > 1:
            ekf_xy = np.array(slam.ekf_trail)
            self.line_ekf.set_data(ekf_xy[:, 0], ekf_xy[:, 1])
            cx, cy = ekf_xy[-1]
            th = slam.ekf.state[2]
            self.pt_ekf.set_data([cx], [cy])
            # Heading arrow
            if self.arrow_ekf is not None:
                self.arrow_ekf.remove()
            self.arrow_ekf = self.ax_traj.annotate(
                '', xy=(cx + 0.18*math.cos(th), cy + 0.18*math.sin(th)),
                xytext=(cx, cy),
                arrowprops=dict(arrowstyle='->', color='green', lw=2.5))

            # Auto-pan view
            pad = PLOT_RADIUS
            self.ax_traj.set_xlim(cx - pad, cx + pad)
            self.ax_traj.set_ylim(cy - pad, cy + pad)

        # Status text
        ex, ey, eth = slam.ekf.state
        ox, oy, oth = slam.odo_x, slam.odo_y, slam.odo_theta
        init_mode = "GEO" if USE_GEOMETRIC_INIT else "ODO"
        self.status_txt.set_text(
            f"Scans: {slam.scan_count}   ICP ok/rej: {slam.icp_accepted}/{slam.icp_rejected}   Init: {init_mode}\n"
            f"EKF  x={ex:.3f} y={ey:.3f} θ={math.degrees(eth):.1f}°\n"
            f"Odo  x={ox:.3f} y={oy:.3f} θ={math.degrees(oth):.1f}°"
        )

        # ── Current scan (sensor frame) ────────────────────────────────────
        if slam.session_log:
            pts = scan_to_xy(slam.session_log[-1])
            if len(pts) > 0:
                self.scat_scan.set_offsets(pts)

        # ── ICP error bar chart ────────────────────────────────────────────
        ax = self.ax_err
        if slam.icp_errors:
            ax.cla()
            ax.set_title("Recent ICP errors", fontsize=10)
            ax.set_xlabel("Pair #"); ax.set_ylabel("Mean error (m)")
            ax.axhline(ICP_ERROR_THRESHOLD, color='red', ls='--', lw=1.2,
                       label=f'Threshold ({ICP_ERROR_THRESHOLD} m)')
            errs = list(slam.icp_errors)
            xs   = list(range(len(errs)))
            cols = ['green' if e <= ICP_ERROR_THRESHOLD else 'red' for e in errs]
            ax.bar(xs, errs, color=cols, alpha=0.8)
            ax.legend(fontsize=8)
            ax.grid(True, ls='--', alpha=0.4)

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def _on_key(self, event):
        if event.key == ' ':
            self.slam.paused = not self.slam.paused
            state = "PAUSED" if self.slam.paused else "RUNNING"
            print(f"[live_slam] {state}")
        elif event.key in ('q', 'Q', 'escape'):
            self.running = False


# ==============================================================================
#  Save session
# ==============================================================================

def save_session(slam: LiveSLAM):
    _here    = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(_here, '..', 'data')
    os.makedirs(data_dir, exist_ok=True)
    ts   = time.strftime("%d_%m_%y_%H_%M_%S")
    mode = "geo" if USE_GEOMETRIC_INIT else "odo"
    fname = os.path.join(data_dir, f"live_slam_{mode}_{ts}.pkl")
    payload = {
        'robot_sensor_signal': slam.session_log,
        'odo_trail': list(slam.odo_trail),
        'ekf_trail': list(slam.ekf_trail),
        'icp_accepted': slam.icp_accepted,
        'icp_rejected': slam.icp_rejected,
        'use_geometric_init': USE_GEOMETRIC_INIT,
    }
    with open(fname, 'wb') as f:
        pickle.dump(payload, f)
    print(f"[live_slam] Session saved → {fname}")
    return fname


# ==============================================================================
#  Main loop
# ==============================================================================

def main():
    print("=" * 60)
    print("  Live SLAM  —  connecting to Arduino")
    print(f"  Local: {LOCAL_IP}:{PORT}   Arduino: {ARDUINO_IP}:{PORT}")
    print(f"  ICP stride: {LIVE_ICP_STRIDE}   Geo init: {USE_GEOMETRIC_INIT}")
    print("  Keys: SPACE=pause  q=quit&save  r=reset")
    print("=" * 60)

    sock = make_socket()
    slam = LiveSLAM()
    plot = LivePlot(slam)

    last_send = time.perf_counter()
    last_plot = time.perf_counter()
    PLOT_INTERVAL = 0.15   # seconds between display refreshes

    print("[live_slam] Waiting for first packet from Arduino...")

    try:
        while plot.running:
            # ── Receive sensor data ───────────────────────────────────────
            result = recv_sensor(sock)
            if result is not None:
                encoder, steering, angles, distances = result
                scan = Scan(encoder, steering, angles, distances)
                if scan.num_lidar_rays >= MIN_SCAN_POINTS:
                    if slam.scan_count == 0:
                        print("[live_slam] First valid scan received. Running.")
                    slam.ingest(scan)

            # ── Send stop command (keeps Arduino happy) ───────────────────
            now = time.perf_counter()
            if now - last_send > SEND_EVERY:
                send_stop(sock)
                last_send = now

            # ── Refresh plot at ~7 Hz ─────────────────────────────────────
            if now - last_plot > PLOT_INTERVAL:
                plot.refresh()
                last_plot = now

            plt.pause(0.001)   # keep GUI event loop alive

    except KeyboardInterrupt:
        print("\n[live_slam] Interrupted.")
    finally:
        send_stop(sock)
        sock.close()
        saved = save_session(slam)
        plt.close('all')
        print(f"[live_slam] Done.  {slam.scan_count} scans, "
              f"{slam.icp_accepted} ICP corrections.")
        print(f"  Final EKF:  ({slam.ekf.state[0]:.3f}, "
              f"{slam.ekf.state[1]:.3f}, "
              f"{math.degrees(slam.ekf.state[2]):.1f} deg)")
        print(f"  Final Odo:  ({slam.odo_x:.3f}, "
              f"{slam.odo_y:.3f}, "
              f"{math.degrees(slam.odo_theta):.1f} deg)")
        print(f"  Saved to:   {saved}")


if __name__ == "__main__":
    main()
