import pickle
import matplotlib.pyplot as plt

filename = "data/test6_robot_data_smooth_right(2)_62_6_04_05_26_01_02_33.pkl"

with open(filename, 'rb') as f:
    data = pickle.load(f)

time = data['time']
encoder = [x.encoder_counts for x in data['robot_sensor_signal']]
steering = [x.steering for x in data['robot_sensor_signal']]
lidar_counts = [x.num_lidar_rays for x in data['robot_sensor_signal']]

plt.figure()
plt.subplot(3,1,1)
plt.plot(time, encoder)
plt.title("Encoder")

plt.subplot(3,1,2)
plt.plot(time, steering)
plt.title("Steering")

plt.subplot(3,1,3)
plt.plot(time, lidar_counts)
plt.title("LIDAR rays")

plt.show()