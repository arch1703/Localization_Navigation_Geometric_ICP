"""
save_results.py
Generates all plots for the 3 best test datasets, for both
USE_GEOMETRIC_INIT = False (odometry init) and True (geometric init).

Outputs go to:
  ../results/geo_off/   — odometry ICP initialisation
  ../results/geo_on/    — geometric line-matching ICP initialisation
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import math, numpy as np, pickle, os

import hybrid_slam as hs

_HERE        = os.path.dirname(os.path.abspath(__file__))
RESULTS_ROOT = os.path.join(_HERE, '..', 'results')
DATA_DIR     = os.path.join(_HERE, '..', 'data')

BEST_TESTS = [
    ('test5', 'test5_robot_data_rough_right_77_20_03_05_26_22_29_29.pkl',    'Rough Right (20 deg)'),
    ('test6', 'test6_robot_data_smooth_right(2)_62_6_04_05_26_01_02_33.pkl', 'Smooth Right (6 deg)'),
    ('test8', 'test8_robot_data_smooth_left_72_-20_03_05_26_18_31_10.pkl',   'Smooth Left (-20 deg)'),
]


def run_and_save(geo_flag, out_subdir):
    hs.USE_GEOMETRIC_INIT = geo_flag
    mode_label = "Geometric Init" if geo_flag else "Odometry Init"
    out_dir = os.path.join(RESULTS_ROOT, out_subdir)
    os.makedirs(out_dir, exist_ok=True)

    for tag, fname, desc in BEST_TESTS:
        print(f"  {tag} [{mode_label}] — {desc}")
        fpath = os.path.join(DATA_DIR, fname)
        with open(fpath, 'rb') as f:
            data = pickle.load(f)
        robot_list = data['robot_sensor_signal']

        ox, oy, oth              = hs.compute_odometry_trajectory(robot_list)
        icp_meas, ix, iy, ith   = hs.compute_icp_trajectory(robot_list)
        ex, ey, eth              = hs.run_ekf_icp(robot_list, icp_meas)
        ekf_poses                = list(zip(ex, ey, eth))
        gpts                     = hs.build_global_map(robot_list, ekf_poses)

        timesteps  = list(range(len(robot_list)))
        icp_t_vals = sorted(icp_meas.keys())
        icp_errs   = [icp_meas[t][3] for t in icp_t_vals]
        n_pairs    = len(icp_meas)

        # ── 1. Trajectory comparison ──────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(8, 7))
        ax.plot(ox, oy, 'b--', lw=1.8, label='Odometry only')
        ax.plot(ix, iy, color='darkorange', lw=1.8, label=f'ICP-chained ({n_pairs} pairs)')
        ax.plot(ex, ey, 'g-',  lw=2.2, label='EKF + ICP')
        ax.plot(ox[0], oy[0], 'ko', ms=9, zorder=5, label='Start')
        ax.plot(ox[-1], oy[-1], 'b^', ms=9, zorder=5,
                label=f'End odo  ({ox[-1]:.2f}, {oy[-1]:.2f}) m')
        ax.plot(ex[-1], ey[-1], 'g^', ms=9, zorder=5,
                label=f'End EKF ({ex[-1]:.2f}, {ey[-1]:.2f}) m')
        for xv, yv, tv, col in [(ox[-1], oy[-1], oth[-1], 'blue'),
                                  (ex[-1], ey[-1], eth[-1], 'green')]:
            ax.annotate('', xy=(xv + 0.12*math.cos(tv), yv + 0.12*math.sin(tv)),
                        xytext=(xv, yv),
                        arrowprops=dict(arrowstyle='->', color=col, lw=2))
        ax.set_title(
            f'{tag}: {desc} — Trajectory Comparison\n'
            f'Init: {mode_label}     {n_pairs} ICP pairs accepted\n'
            f'Odometry final: ({ox[-1]:.3f}, {oy[-1]:.3f}, {math.degrees(oth[-1]):.1f} deg)   '
            f'EKF final: ({ex[-1]:.3f}, {ey[-1]:.3f}, {math.degrees(eth[-1]):.1f} deg)'
        )
        ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)')
        ax.set_aspect('equal'); ax.grid(True, ls='--', alpha=0.5)
        ax.legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f'{tag}_trajectory.png'), dpi=150)
        plt.close()

        # ── 2. Geometric environment map ──────────────────────────────────────
        fig, ax = plt.subplots(figsize=(8, 7))
        if len(gpts) > 0:
            ax.scatter(gpts[:, 0], gpts[:, 1], s=1, c='gray', alpha=0.25,
                       label=f'LiDAR map ({len(gpts)} pts)')
        ax.plot(ox, oy, 'b--', lw=1.2, alpha=0.7, label='Odometry')
        ax.plot(ex, ey, 'g-',  lw=2.0, label='EKF trajectory')
        ax.plot(ex[0], ey[0], 'ko', ms=9, label='Start')
        ax.set_title(f'{tag}: {desc} — Geometric Environment Map\nInit: {mode_label}')
        ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)')
        ax.set_aspect('equal'); ax.grid(True, ls='--', alpha=0.5)
        ax.legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f'{tag}_map.png'), dpi=150)
        plt.close()

        # ── 3. Heading over time ──────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(timesteps, [math.degrees(t) for t in oth], 'b--', lw=1.5, label='Odometry theta')
        ax.plot(timesteps, [math.degrees(t) for t in eth], 'g-',  lw=1.8, label='EKF theta')
        if icp_t_vals:
            icp_th_deg = [math.degrees(icp_meas[t][2]) for t in icp_t_vals]
            ax.scatter(icp_t_vals, icp_th_deg, c='orange', s=30, zorder=5,
                       label='ICP theta measurements')
        ax.set_title(f'{tag}: {desc} — Heading over Time ({mode_label})')
        ax.set_xlabel('Timestep'); ax.set_ylabel('Heading (degrees)')
        ax.grid(True, ls='--', alpha=0.5); ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f'{tag}_heading.png'), dpi=150)
        plt.close()

        # ── 4. Per-pair ICP error bar chart ──────────────────────────────────
        fig, ax = plt.subplots(figsize=(9, 4))
        if icp_t_vals:
            ax.bar(icp_t_vals, icp_errs, color='steelblue', alpha=0.85,
                   label='ICP mean error (m)')
            ax.axhline(hs.ICP_ERROR_THRESHOLD, color='red', ls='--', lw=1.5,
                       label=f'Threshold ({hs.ICP_ERROR_THRESHOLD} m)')
            ax.set_xlim(0, len(robot_list))
        else:
            ax.text(0.5, 0.5, 'No ICP pairs accepted', transform=ax.transAxes,
                    ha='center', va='center', fontsize=14, color='gray')
        ax.set_title(
            f'{tag}: {desc} — Per-pair ICP Error ({mode_label}, {n_pairs} accepted)')
        ax.set_xlabel('Timestep index'); ax.set_ylabel('Mean point-to-point error (m)')
        ax.grid(True, ls='--', alpha=0.5); ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f'{tag}_icp_errors.png'), dpi=150)
        plt.close()

        # ── 5. State components (X, Y, theta) over time ──────────────────────
        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        panels = [
            (ox, ex, [icp_meas[t][0] for t in icp_t_vals], 'X (m)'),
            (oy, ey, [icp_meas[t][1] for t in icp_t_vals], 'Y (m)'),
            ([math.degrees(v) for v in oth],
             [math.degrees(v) for v in eth],
             [math.degrees(icp_meas[t][2]) for t in icp_t_vals],
             'Heading (deg)'),
        ]
        for ax, (vals_odo, vals_ekf, vals_icp, ylabel) in zip(axes, panels):
            ax.plot(timesteps, vals_odo, 'b--', lw=1.5, label='Odometry')
            ax.plot(timesteps, vals_ekf, 'g-',  lw=1.8, label='EKF')
            if icp_t_vals:
                ax.scatter(icp_t_vals, vals_icp, c='orange', s=20, zorder=5,
                           label='ICP meas.')
            ax.set_ylabel(ylabel); ax.grid(True, ls='--', alpha=0.5)
        axes[0].legend(fontsize=8)
        axes[2].set_xlabel('Timestep')
        fig.suptitle(
            f'{tag}: {desc} — State Components over Time ({mode_label})', fontsize=12)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f'{tag}_state_components.png'), dpi=150)
        plt.close()

        # ── 6. ICP vs odometry displacement per pair ──────────────────────────
        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        if icp_t_vals and len(icp_t_vals) > 1:
            # ICP displacement magnitudes between consecutive ICP poses
            icp_ds = [math.hypot(icp_meas[icp_t_vals[k]][0] - icp_meas[icp_t_vals[k-1]][0],
                                  icp_meas[icp_t_vals[k]][1] - icp_meas[icp_t_vals[k-1]][1])
                      for k in range(1, len(icp_t_vals))]
            odo_ds_at_icp = []
            for k in range(1, len(icp_t_vals)):
                ta, tb = icp_t_vals[k-1], icp_t_vals[k]
                odo_d = math.hypot(ox[tb] - ox[ta], oy[tb] - oy[ta])
                odo_ds_at_icp.append(odo_d)
            axes[0].bar(range(len(icp_ds)), odo_ds_at_icp, alpha=0.6, label='Odometry stride dist (m)', color='blue')
            axes[0].bar(range(len(icp_ds)), icp_ds, alpha=0.6, label='ICP stride dist (m)', color='orange')
            axes[0].set_ylabel('Displacement (m)')
            axes[0].legend(fontsize=9); axes[0].grid(True, ls='--', alpha=0.5)

            icp_dth = [math.degrees(hs.wrap_angle(
                            icp_meas[icp_t_vals[k]][2] - icp_meas[icp_t_vals[k-1]][2]))
                        for k in range(1, len(icp_t_vals))]
            odo_dth = [math.degrees(hs.wrap_angle(oth[icp_t_vals[k]] - oth[icp_t_vals[k-1]]))
                        for k in range(1, len(icp_t_vals))]
            axes[1].plot(range(len(odo_dth)), odo_dth, 'b--o', ms=5, lw=1.5, label='Odometry dtheta (deg)')
            axes[1].plot(range(len(icp_dth)), icp_dth, 'orange', marker='o', ms=5, lw=1.5, label='ICP dtheta (deg)')
            axes[1].set_ylabel('Heading change (deg)')
            axes[1].set_xlabel('ICP pair index')
            axes[1].legend(fontsize=9); axes[1].grid(True, ls='--', alpha=0.5)
        else:
            for ax in axes:
                ax.text(0.5, 0.5, 'Insufficient ICP pairs', transform=ax.transAxes,
                        ha='center', va='center', fontsize=12, color='gray')
        fig.suptitle(
            f'{tag}: {desc} — ICP vs Odometry per stride ({mode_label})', fontsize=12)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f'{tag}_icp_vs_odo.png'), dpi=150)
        plt.close()

        print(f"    Odo:({ox[-1]:.3f},{oy[-1]:.3f},{math.degrees(oth[-1]):.1f}deg)  "
              f"ICP:({ix[-1]:.3f},{iy[-1]:.3f},{math.degrees(ith[-1]):.1f}deg)  "
              f"EKF:({ex[-1]:.3f},{ey[-1]:.3f},{math.degrees(eth[-1]):.1f}deg)  "
              f"pairs={n_pairs}")

    print(f"  -> Saved 6 plots x {len(BEST_TESTS)} tests = {6*len(BEST_TESTS)} files to {out_dir}")


print("=" * 60)
print("  Saving results for 3 best tests, both ICP init modes")
print("=" * 60)

print("\n[Pass 1] USE_GEOMETRIC_INIT = False  (odometry init)")
run_and_save(False, 'geo_off')

print("\n[Pass 2] USE_GEOMETRIC_INIT = True   (geometric line matching)")
run_and_save(True, 'geo_on')

print("\nDone. All plots saved.")
