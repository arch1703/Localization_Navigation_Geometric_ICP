import pickle
import math
import matplotlib.pyplot as plt


files = [
    "./data/test3_robot_data_smooth_straight(2)_61_0_04_05_26_00_55_11.pkl",
    "./data/test10_robot_data_smooth_left(2)57_-12_04_05_26_01_09_54.pkl",
    "./data/test6_robot_data_smooth_right(2)_62_6_04_05_26_01_02_33.pkl",
    "./data/test9_robot_data_rough_left_67_-18_03_05_26_22_39_07.pkl",
    "./data/test5_robot_data_rough_right_77_20_03_05_26_22_29_29.pkl",
]

labels = [
    "Smooth straight",
    "Smooth left",
    "Smooth right",
    "Rough left",
    "Rough right",
]


wheel_radius = 0.034
encoder_counts_per_rev = 152
wheel_base = 0.15


def encoder_counts_to_distance(delta_counts):
    return 2 * math.pi * wheel_radius * delta_counts / encoder_counts_per_rev


def wrap_angle(theta):
    return math.atan2(math.sin(theta), math.cos(theta))


def compute_trajectory(filename):
    with open(filename, "rb") as f:
        data = pickle.load(f)

    time_list = data["time"]
    robot_list = data["robot_sensor_signal"]

    t0 = time_list[0]
    time_list = [t - t0 for t in time_list]

    x = 0.0
    y = 0.0
    theta = 0.0

    x_list = [x]
    y_list = [y]

    last_encoder = robot_list[0].encoder_counts

    for i in range(1, len(robot_list)):
        encoder = robot_list[i].encoder_counts
        steering_deg = robot_list[i].steering

        delta_counts = encoder - last_encoder
        last_encoder = encoder

        ds = encoder_counts_to_distance(delta_counts)
        steering_rad = math.radians(steering_deg)

        dtheta = (ds / wheel_base) * math.tan(steering_rad)

        theta = wrap_angle(theta + dtheta)

        x += ds * math.cos(theta)
        y += ds * math.sin(theta)

        x_list.append(x)
        y_list.append(y)

    return x_list, y_list


plt.figure(figsize=(8, 6))

for file, label in zip(files, labels):
    try:
        x_list, y_list = compute_trajectory(file)
        plt.plot(x_list, y_list, label=label)
    except FileNotFoundError:
        print("File not found:", file)

plt.xlabel("X (m)")
plt.ylabel("Y (m)")
plt.title("Odometry-only trajectories")
plt.axis("equal")
plt.grid(True)
plt.legend()
plt.show()