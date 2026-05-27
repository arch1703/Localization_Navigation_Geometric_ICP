import pickle
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree


filename = "./data/test6_robot_data_smooth_right(2)_62_6_04_05_26_01_02_33.pkl"

idx1 = 46
idx2 = 50

encoder_scale = 0.00026457
wheel_base = 0.15


def scan_to_xy(scan):
    points = []

    for angle_deg, distance_mm in zip(scan.angles, scan.distances):
        if distance_mm <= 20:
            continue

        distance_m = distance_mm / 1000.0
        angle_rad = -math.radians(angle_deg)

        x = distance_m * math.cos(angle_rad)
        y = distance_m * math.sin(angle_rad)

        if abs(x) < 5 and abs(y) < 5:
            points.append([x, y])

    return np.array(points)


def rotate_points(points, theta):
    R = np.array([
        [math.cos(theta), -math.sin(theta)],
        [math.sin(theta),  math.cos(theta)]
    ])
    return points @ R.T


def icp(source, target, max_iters=20, tolerance=1e-5, max_match_distance=0.4):
    src = source.copy()
    dst = target.copy()

    prev_error = float("inf")

    total_R = np.eye(2)
    total_t = np.zeros(2)

    for _ in range(max_iters):
        tree = cKDTree(dst)
        distances, indices = tree.query(src)

        mask = distances < max_match_distance

        if np.sum(mask) < 5:
            print("Too few valid ICP matches")
            break

        src_valid = src[mask]
        matched_dst = dst[indices[mask]]

        src_centroid = np.mean(src_valid, axis=0)
        dst_centroid = np.mean(matched_dst, axis=0)

        src_centered = src_valid - src_centroid
        dst_centered = matched_dst - dst_centroid

        H = src_centered.T @ dst_centered

        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T

        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        t = dst_centroid - R @ src_centroid

        src = (R @ src.T).T + t

        total_R = R @ total_R
        total_t = R @ total_t + t

        mean_error = np.mean(distances[mask])

        if abs(prev_error - mean_error) < tolerance:
            break

        prev_error = mean_error

    return src, total_R, total_t, prev_error


with open(filename, "rb") as f:
    data = pickle.load(f)

robot_list = data["robot_sensor_signal"]

scan1 = scan_to_xy(robot_list[idx1])
scan2 = scan_to_xy(robot_list[idx2])

encoder_diff = robot_list[idx2].encoder_counts - robot_list[idx1].encoder_counts
dx = encoder_diff * encoder_scale

steering_deg = robot_list[idx1].steering
steering_rad = math.radians(steering_deg)

# Odometry-based rotation estimate
dtheta = (dx / wheel_base) * math.tan(steering_rad)

print("Encoder diff:", encoder_diff)
print("dx:", dx)
print("Steering deg:", steering_deg)
print("Initial dtheta rad:", dtheta)
print("Initial dtheta deg:", math.degrees(dtheta))

# IMPORTANT:
# Try changing sign if needed:
# scan1_initial = rotate_points(scan1, dtheta) + np.array([dx, 0.0])
dx_scaled = 0.6 * dx
scan1_initial = rotate_points(scan1, dtheta) + np.array([dx, 0.0])

aligned_scan1, R_icp, t_icp, error = icp(scan1_initial, scan2)

theta_icp = math.atan2(R_icp[1, 0], R_icp[0, 0])

print("\nICP Result")
print("R_icp:")
print(R_icp)
print("t_icp:", t_icp)
print("theta_icp rad:", theta_icp)
print("theta_icp deg:", math.degrees(theta_icp))
print("Mean ICP error:", error)

plt.figure(figsize=(7, 7))
plt.scatter(scan2[:, 0], scan2[:, 1], s=12, label=f"Target scan {idx2}")
plt.scatter(scan1[:, 0], scan1[:, 1], s=12, label=f"Original scan {idx1}")
plt.scatter(scan1_initial[:, 0], scan1_initial[:, 1], s=12, label="Odometry initial guess")
plt.scatter(aligned_scan1[:, 0], aligned_scan1[:, 1], s=12, label="ICP aligned scan")

plt.plot(0, 0, "ro", label="Robot")
plt.xlabel("X (m)")
plt.ylabel("Y (m)")
plt.title("ICP with Translation + Rotation Initial Guess")
plt.axis("equal")
plt.grid(True)
plt.legend()
plt.show()