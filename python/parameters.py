# External libraries
import math
import numpy as np

# UDP parameters
localIP = "192.168.0.200" # Put your laptop computer's IP here 199 / Arnav198
arduinoIP = "192.168.0.197" # Put your arduino's IP here 200
localPort = 4010
arduinoPort = 4010
bufferSize = 8192

# Camera parameters
camera_id = 0
marker_length = 0.0762
camera_matrix = np.array([[1.52370613e+03, 0.00000000e+00, 7.30421569e+02],
                            [0.00000000e+00, 1.52405762e+03, 4.57716793e+02],
                            [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]], dtype=np.float32)
dist_coeffs = np.array([-3.66501798e-01,  3.81995311e-02, -2.98211989e-05,  1.99327127e-05, 8.98904526e-02], dtype=np.float32)


# Robot parameters
num_robot_sensors = 2 # encoder, steering
num_robot_control_signals = 2 # speed, steering
L = 0.15 # wheel base 15cm

# Logging parameters
max_num_lines_before_write = 1
filename_start = './data/robot_data'
data_name_list = ['time', 'control_signal', 'robot_sensor_signal', 'camera_sensor_signal', 'state_mean', 'state_covariance']

# Experiment trial parameters
trial_time = 30000 # milliseconds
extra_trial_log_time = 2000 # milliseconds

# KF parameters
I3 = np.array([[1, 0, 0],[0, 1, 0], [0, 0, 1]])
covariance_plot_scale = 100