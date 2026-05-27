import pickle
import math
import matplotlib.pyplot as plt


filename = "./data/test3_robot_data_smooth_straight(2)_61_0_04_05_26_00_55_11.pkl"

scan_index = 50   # change this if needed


with open(filename, "rb") as f:
    data = pickle.load(f)

robot_list = data["robot_sensor_signal"]

scan = robot_list[scan_index]

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

plt.figure(figsize=(6, 6))
plt.scatter(x_points, y_points, s=10)
plt.plot(0, 0, "ro", label="Robot")
plt.xlabel("X (m)")
plt.ylabel("Y (m)")
plt.title("Single LIDAR scan")
plt.axis("equal")
plt.grid(True)
plt.legend()
plt.show()

print("Number of LIDAR rays:", scan.num_lidar_rays)
print("First 10 angles:", scan.angles[:10])
print("First 10 distances:", scan.distances[:10])