import pickle
import math
import matplotlib.pyplot as plt


# CHANGE FILE
filename = "./data/test6_robot_data_smooth_right(2)_62_6_04_05_26_01_02_33.pkl"

# Choose two different time steps
idx1 = 50
idx2 = 46


def scan_to_xy(scan):
    x_points = []
    y_points = []

    for angle_deg, distance_mm in zip(scan.angles, scan.distances):
        if distance_mm <= 20:
            continue

        distance_m = distance_mm / 1000.0
        angle_rad = -math.radians(angle_deg)

        x = distance_m * math.cos(angle_rad)
        y = distance_m * math.sin(angle_rad)

        if abs(x) < 5 and abs(y) < 5:
            x_points.append(x)
            y_points.append(y)

    return x_points, y_points


# Load data
with open(filename, "rb") as f:
    data = pickle.load(f)

robot_list = data["robot_sensor_signal"]

scan1 = robot_list[idx1]
scan2 = robot_list[idx2]

# Convert to XY
x1, y1 = scan_to_xy(scan1)
x2, y2 = scan_to_xy(scan2)

# Plot
plt.figure(figsize=(6,6))
plt.scatter(x1, y1, s=10, label=f"Scan {idx1}")
plt.scatter(x2, y2, s=10, label=f"Scan {idx2}")

plt.plot(0, 0, "ro", label="Robot")

plt.xlabel("X (m)")
plt.ylabel("Y (m)")
plt.title("Two LIDAR scans (before alignment)")
plt.axis("equal")
plt.grid(True)
plt.legend()
plt.show()