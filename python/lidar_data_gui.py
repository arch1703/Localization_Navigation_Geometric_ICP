import math
import time
import os
import numpy as np
import matplotlib.pyplot as plt
from nicegui import ui

import parameters
import robot_python_code


class SimpleLidarGUI:
    def __init__(self):
        self.connected = False
        self.udp = None

        self.robot_sensor_signal = robot_python_code.RobotSensorSignal([0, 0, 0])
        self.data_logger = robot_python_code.DataLogger(
            parameters.filename_start,
            parameters.data_name_list
        )

        self.control_signal = [0, 0]

        self.max_lidar_range = 5.0
        self.lidar_angle_res = 2
        self.num_angles = int(360 / self.lidar_angle_res)
        self.lidar_distance_list = [self.max_lidar_range for _ in range(self.num_angles)]

        self.running_test = False
        self.test_start_time = 0
        self.test_duration = 10.0
        self.test_number = 1

        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.last_encoder = 0
        self.last_time = time.perf_counter()
        self.path_points = [[0, 0]]

    def connect_robot(self):
        if not self.connected:
            self.udp, success = robot_python_code.create_udp_communication(
                parameters.arduinoIP,
                parameters.localIP,
                parameters.arduinoPort,
                parameters.localPort,
                parameters.bufferSize,
            )
            self.connected = success
            print("Robot connected" if success else "Robot connection failed")

    def disconnect_robot(self):
        self.connected = False
        self.udp = None
        print("Robot disconnected")

    def start_test(self):
        if not self.connected:
            print("Connect robot first.")
            return

        self.running_test = True
        self.test_start_time = time.perf_counter()
        self.reset_path()

        test_name = f"test{self.test_number}"
        self.data_logger.filename_start = f"./data/{test_name}_robot_data"

        print(f"10 second test started: {test_name}")

    def reset_path(self):
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.last_encoder = self.robot_sensor_signal.encoder_counts
        self.last_time = time.perf_counter()
        self.path_points = [[0, 0]]

    def parse_robot_message(self, msg):
        try:
            if msg == "":
                return

            values = [float(x.strip()) for x in msg.split(",") if x.strip() != ""]

            if len(values) < 3:
                return

            num_lidar_rays = int(values[2])
            expected_len = 3 + 2 * num_lidar_rays

            if len(values) < expected_len:
                print("Incomplete LIDAR packet:", len(values), "expected:", expected_len)
                return

            self.robot_sensor_signal = robot_python_code.RobotSensorSignal(values)
            self.update_lidar_distances()
            self.update_odometry()

        except Exception as e:
            print("Parse error:", e)

    def receive_robot_data(self):
        if self.connected and self.udp is not None:
            msg = self.udp.receive_msg()
            self.parse_robot_message(msg)

    def send_control(self, speed, steering):
        self.control_signal = [speed, steering]

        if self.connected and self.udp is not None:
            self.udp.send_msg(f"{speed}, {steering}\n")

    def log_data(self, logging_on):
        self.data_logger.log(
            logging_on,
            time.perf_counter(),
            self.control_signal,
            self.robot_sensor_signal,
            [0, 0, 0, 0, 0, 0],
            np.array([self.x, self.y, self.theta]),
            np.eye(3),
        )

    def update_odometry(self):
        now = time.perf_counter()
        self.last_time = now

        encoder = self.robot_sensor_signal.encoder_counts
        delta_counts = encoder - self.last_encoder
        self.last_encoder = encoder

        delta_s = 0.00026457 * delta_counts

        steering_deg = self.control_signal[1]
        steering_rad = math.radians(steering_deg)

        L = 0.15

        if abs(delta_s) > 0.000001:
            delta_theta = -(delta_s / L) * math.tan(steering_rad)

            self.theta += delta_theta
            self.theta = math.atan2(math.sin(self.theta), math.cos(self.theta))

            self.x += delta_s * math.cos(self.theta)
            self.y += delta_s * math.sin(self.theta)

            self.path_points.append([self.x, self.y])

    def update_lidar_distances(self):
        for angle_deg, distance_mm in zip(
            self.robot_sensor_signal.angles,
            self.robot_sensor_signal.distances
        ):
            if distance_mm <= 20:
                continue

            distance_m = distance_mm / 1000.0
            distance_m = min(distance_m, self.max_lidar_range)

            angle = 360 - angle_deg

            if 0 <= angle < 360:
                index = int(angle / self.lidar_angle_res)
                index = max(0, min(self.num_angles - 1, index))
                self.lidar_distance_list[index] = distance_m

    def get_lidar_ray_lines(self):
        rays = []

        for i in range(self.num_angles):
            angle_deg = i * self.lidar_angle_res
            angle_rad = math.radians(angle_deg)

            hit_distance = self.lidar_distance_list[i]

            x1 = hit_distance * math.cos(angle_rad)
            y1 = hit_distance * math.sin(angle_rad)

            x2 = self.max_lidar_range * math.cos(angle_rad)
            y2 = self.max_lidar_range * math.sin(angle_rad)

            rays.append({"coords": [[x1, y1], [x2, y2]]})

        return rays

    def get_hit_points(self):
        points = []

        for i in range(self.num_angles):
            distance = self.lidar_distance_list[i]

            if distance >= self.max_lidar_range:
                continue

            angle_deg = i * self.lidar_angle_res
            angle_rad = math.radians(angle_deg)

            x = distance * math.cos(angle_rad)
            y = distance * math.sin(angle_rad)

            points.append([x, y])

        return points

    def clear_lidar_map(self):
        self.lidar_distance_list = [self.max_lidar_range for _ in range(self.num_angles)]

    def save_path_graph(self):
        os.makedirs("data", exist_ok=True)

        filename = f"data/test{self.test_number}_path_graph_{time.strftime('%Y%m%d_%H%M%S')}.png"
        path = np.array(self.path_points)

        plt.figure()
        plt.plot(path[:, 0], path[:, 1], "bo-")
        plt.plot(path[0, 0], path[0, 1], "go", label="Start")
        plt.plot(path[-1, 0], path[-1, 1], "ro", label="End")
        plt.xlabel("X position (m)")
        plt.ylabel("Y position (m)")
        plt.title(f"Test {self.test_number}: Estimated Robot Path")
        plt.grid(True)
        plt.axis("equal")
        plt.legend()
        plt.savefig(filename)
        plt.close()

        print(f"Path graph saved: {filename}")


robot = SimpleLidarGUI()

ui.dark_mode().enable()

with ui.card().classes("w-full items-center"):
    ui.label("ROB-GY 6213: LIDAR Map + Robot Path + 10s Test Logger").style("font-size: 24px")

with ui.row().classes("w-full"):
    with ui.card().classes("w-1/2"):
        ui.label("LIDAR Map")
        lidar_chart = ui.echart({
            "backgroundColor": "#000000",
            "animation": False,
            "xAxis": {"min": -2, "max": 2, "name": "X (m)"},
            "yAxis": {"min": -2, "max": 2, "name": "Y (m)"},
            "series": [
                {
                    "name": "LIDAR Rays",
                    "type": "lines",
                    "coordinateSystem": "cartesian2d",
                    "data": [],
                    "lineStyle": {"color": "#ff3333", "width": 1, "opacity": 0.85},
                    "effect": {"show": False},
                    "animation": False,
                },
                {
                    "name": "Hit Points",
                    "type": "scatter",
                    "data": [],
                    "symbolSize": 5,
                    "itemStyle": {"color": "#00ff66"},
                    "animation": False,
                },
                {
                    "name": "Robot",
                    "type": "scatter",
                    "data": [[0, 0]],
                    "symbolSize": 12,
                    "itemStyle": {"color": "#3399ff"},
                    "animation": False,
                },
            ],
        }).classes("w-full h-[550px]")

    with ui.card().classes("w-1/2"):
        ui.label("Robot Path")
        path_chart = ui.echart({
            "backgroundColor": "#000000",
            "animation": False,
            "xAxis": {"min": -2, "max": 2, "name": "X (m)"},
            "yAxis": {"min": -2, "max": 2, "name": "Y (m)"},
            "series": [
                {
                    "name": "Path",
                    "type": "line",
                    "data": [],
                    "lineStyle": {"color": "#00ccff", "width": 3},
                    "symbolSize": 5,
                    "animation": False,
                },
                {
                    "name": "Robot Position",
                    "type": "scatter",
                    "data": [[0, 0]],
                    "symbolSize": 12,
                    "itemStyle": {"color": "#ffcc00"},
                    "animation": False,
                },
            ],
        }).classes("w-full h-[550px]")

with ui.card().classes("w-full"):
    with ui.row():
        encoder_label = ui.label("Encoder: 0")
        lidar_label = ui.label("LIDAR rays: 0")
        status_label = ui.label("Status: Not connected")
        test_label = ui.label("Test: Not running")
        pose_label = ui.label("Pose: x=0.00 y=0.00 θ=0.0°")
        test_number_label = ui.label("Next file: test1")

    connect_switch = ui.switch("Robot Connect")

    ui.label("Speed")
    speed_slider = ui.slider(min=0, max=100, value=0)

    ui.label("Steering")
    steering_slider = ui.slider(min=-20, max=20, value=0)

    speed_enable = ui.switch("Enable Speed")
    steering_enable = ui.switch("Enable Steering")

    def run_test_button_clicked():
        speed_enable.value = True
        steering_enable.value = True
        robot.start_test()

    ui.button("Run Test 10s", on_click=run_test_button_clicked)
    ui.button("Clear LIDAR Map", on_click=robot.clear_lidar_map)
    ui.button("Reset Path", on_click=robot.reset_path)


async def update_loop():
    if connect_switch.value and not robot.connected:
        robot.connect_robot()
    elif not connect_switch.value and robot.connected:
        robot.disconnect_robot()

    speed = int(speed_slider.value) if speed_enable.value else 0
    steering = int(steering_slider.value) if steering_enable.value else 0

    robot.receive_robot_data()
    robot.send_control(speed, steering)

    logging_now = False

    if robot.running_test:
        elapsed = time.perf_counter() - robot.test_start_time

        if elapsed <= robot.test_duration:
            logging_now = True
            test_label.set_text(f"Test {robot.test_number}: Running {elapsed:.1f}/10.0 s")
        else:
            logging_now = False
            robot.running_test = False

            speed_enable.value = False
            steering_enable.value = False

            test_label.set_text(f"Test {robot.test_number}: Finished")
            robot.save_path_graph()
            print(f"Test {robot.test_number} finished. Data and path graph saved.")

            robot.test_number += 1
            test_number_label.set_text(f"Next file: test{robot.test_number}")
    else:
        test_label.set_text("Test: Not running")

    robot.log_data(logging_now)

    encoder_label.set_text(f"Encoder: {robot.robot_sensor_signal.encoder_counts}")
    lidar_label.set_text(f"LIDAR rays: {robot.robot_sensor_signal.num_lidar_rays}")
    status_label.set_text("Status: Connected" if robot.connected else "Status: Not connected")
    pose_label.set_text(f"Pose: x={robot.x:.2f} y={robot.y:.2f} θ={math.degrees(robot.theta):.1f}°")
    test_number_label.set_text(f"Next file: test{robot.test_number}")

    lidar_chart.options["series"][0]["data"] = robot.get_lidar_ray_lines()
    lidar_chart.options["series"][1]["data"] = robot.get_hit_points()
    lidar_chart.update()

    path_chart.options["series"][0]["data"] = robot.path_points
    path_chart.options["series"][1]["data"] = [[robot.x, robot.y]]
    path_chart.update()


ui.timer(0.1, update_loop)

ui.run(native=True)