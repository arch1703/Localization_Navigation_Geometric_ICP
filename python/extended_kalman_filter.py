# External libraries
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import csv

# Local libraries
import parameters
import data_handling
from motion_models import *

# Main class
class ExtendedKalmanFilter:
    def __init__(self, x_0, Sigma_0, encoder_counts_0):
        self.state_mean = x_0
        self.state_covariance = Sigma_0
        self.predicted_state_mean = [0,0,0]
        self.predicted_state_covariance = parameters.I3 * 1.0
        self.last_encoder_counts = encoder_counts_0

    # Call the prediction and correction steps
    def update(self, u_t, z_t, delta_t, camera_sees_tag):
        self.prediction_step(u_t, delta_t)
        self.state_mean = self.predicted_state_mean
        self.state_covariance = self.predicted_state_covariance
        if camera_sees_tag:
            self.correction_step(z_t)
        return

    # Set the EKF's predicted state mean and covariance matrix
    def prediction_step(self, u_t, delta_t):
        x_pred, delta_s, delta_theta = self.g_function(self.state_mean, u_t, delta_t)
        G_x = self.get_G_x(self.state_mean, delta_s, delta_theta)
        G_u = self.get_G_u(self.state_mean, delta_s, u_t[1])
        R = self.get_R(delta_s, u_t[1])
        Sigma_pred = (
            G_x @ self.state_covariance @ G_x.T +
            G_u @ R @ G_u.T
        )
        self.predicted_state_mean = x_pred
        self.predicted_state_covariance = Sigma_pred

    # Set the EKF's corrected state mean and covariance matrix
    def correction_step(self, z_t):
        H = self.get_H()
        Q = self.get_Q()

        z_pred = self.get_h_function(self.predicted_state_mean)

        y = z_t - z_pred 
        y[2] = math.atan2(math.sin(y[2]), math.cos(y[2]))  

        S = H @ self.predicted_state_covariance @ H.T + Q
        K = self.predicted_state_covariance @ H.T @ np.linalg.inv(S)

        self.state_mean = self.predicted_state_mean + K @ y

        I = parameters.I3
        self.state_covariance = (I - K @ H) @ self.predicted_state_covariance
        
        return

    # Function to calculate distance from encoder counts
    def distance_travelled_s(self, encoder_counts):
        s = 0.00026457*encoder_counts #+ 0.059503
        return s    
            
    # Function to calculate rotational velocity from steering and dist travelled or speed
    def rotational_velocity_w(self, steering_angle_command): 
        #w = 0.56429*steering_angle_command + -3.3714
        w = -0.56429*steering_angle_command + 3.3714
        w = math.radians(w)
        return w

    # The nonlinear transition equation that provides new states from past states
    def g_function(self, x_tm1, u_t, delta_t):  
            encoder_counts = u_t[0]
            phi = u_t[1]
            phi = math.radians(phi)
            L = parameters.L

            delta_counts = encoder_counts - self.last_encoder_counts
            self.last_encoder_counts = encoder_counts

            delta_s = self.distance_travelled_s(delta_counts)
            # bicycle
            delta_theta = -(delta_s / L) * math.tan(phi)

            theta = x_tm1[2] + delta_theta
            theta = math.atan2(math.sin(theta), math.cos(theta)) 
            x_new = x_tm1[0] + delta_s * math.cos(theta)
            y_new = x_tm1[1] + delta_s * math.sin(theta)
            return np.array([x_new, y_new, theta]), delta_s, delta_theta
        
    
    # The nonlinear measurement function
    def get_h_function(self, x_t):
        return x_t
    
    # This function returns a matrix with the partial derivatives dg/dx
    # g outputs x_t, y_t, theta_t, and we take derivatives wrt inputs x_tm1, y_tm1, theta_tm1
    def get_G_x(self, x, delta_s, delta_theta):      
        theta = x[2]
        theta_mid = theta + delta_theta/2
        G = np.array([
            [1, 0, -delta_s * math.sin(theta_mid)],
            [0, 1,  delta_s * math.cos(theta_mid)],
            [0, 0, 1]
        ])
        return G

    # This function returns a matrix with the partial derivatives dg/du
    def get_G_u(self, x, delta_s, phi):             
        theta = x[2]
        L = parameters.L
        G_u = np.array([
            [math.cos(theta), 0],
            [math.sin(theta), 0],
            [(1/L)*math.tan(phi), (delta_s/L)*(1/(math.cos(phi)**2))]
        ])
        return G_u

    # This function returns a matrix with the partial derivatives dh_t/dx_t
    def get_H(self):
        return parameters.I3
    
    # This function returns the R_t matrix which contains transition function covariance terms.
    def get_R(self, s, phi):
        var_s = variance_distance_travelled_s(s)
        var_w = variance_rotational_velocity_w(phi)
        R = np.array([
            [var_s, 0],
            [0, var_w]
        ])
        return R

    # This function returns the Q_t matrix which contains measurement covariance terms.
    def get_Q(self):
        var_x = 1.8393
        var_y = 1.5536
        var_theta = 0.0004
        Q = np.array([
            [var_x, 0, 0],
            [0, var_y, 0],
            [0, 0, var_theta]
        ])
        return Q

class KalmanFilterPlot:

    def __init__(self):
        self.dir_length = 0.1
        fig, ax = plt.subplots()
        self.ax = ax
        self.fig = fig

        # Storage for history
        self.history_x = []
        self.history_y = []
        self.history_theta = []
        self.history_cov = []

    def update(self, state_mean, state_covariance):
        self.history_x.append(state_mean[0])
        self.history_y.append(state_mean[1])
        self.history_theta.append(state_mean[2])
        self.history_cov.append(state_covariance)
        plt.clf()

        # Plot covariance ellipse
        lambda_, v = np.linalg.eig(state_covariance)
        lambda_ = np.sqrt(lambda_)
        xy = (state_mean[0], state_mean[1])
        angle=np.rad2deg(np.arctan2(*v[:,0][::-1]))
        ell = Ellipse(xy, alpha=0.5, facecolor='red',width=lambda_[0], height=lambda_[1], angle = angle)
        ax = self.fig.gca()
        ax.add_artist(ell)
        
        # Plot state estimate
        plt.plot(state_mean[0], state_mean[1],'ro')
        plt.plot([state_mean[0], state_mean[0]+ self.dir_length*math.cos(state_mean[2]) ], [state_mean[1], state_mean[1]+ self.dir_length*math.sin(state_mean[2]) ],'r')
        plt.xlabel('X(m)')
        plt.ylabel('Y(m)')
        plt.axis([-1, 1, -1, 1])
        plt.grid()
        plt.draw()
        plt.pause(0.1)

    def plot_uncertainty_history(self):
        uncertainty_trace = [np.trace(cov) for cov in self.history_cov]
        std_x = [np.sqrt(cov[0, 0]) for cov in self.history_cov]
        std_y = [np.sqrt(cov[1, 1]) for cov in self.history_cov]
        time_steps = range(len(uncertainty_trace))

        plt.figure(figsize=(10, 6))
        
        # trace
        plt.subplot(2, 1, 1)
        plt.plot(time_steps, uncertainty_trace, 'r-', linewidth=2, label=r'Trace of Covariance ($\sigma$)')
        plt.ylabel('Total Variance ($m^2 + rad^2$)')
        plt.title('Total Filter Uncertainty Over Time')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()

        # standard deviations
        plt.subplot(2, 1, 2)
        plt.plot(time_steps, std_x, 'b-', label=r'$\sigma_x$ (Position X)')
        plt.plot(time_steps, std_y, 'g-', label=r'$\sigma_y$ (Position Y)')
        plt.xlabel('Time Step')
        plt.ylabel('Standard Deviation (m)')
        plt.title(r'Positional Uncertainty, $\sigma$')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()

        plt.tight_layout()
        plt.show()

    def save_and_plot_final(self):
        plt.figure()
        ax = plt.gca()
        # Plot the full line
        plt.plot(self.history_x, self.history_y, 'b-', label='EKF Trajectory')
        for i in range(0, len(self.history_x), 20):
            mean = [self.history_x[i], self.history_y[i], self.history_theta[i]]
            cov = self.history_cov[i][0:2, 0:2] 
            lambda_, v = np.linalg.eig(cov)
            lambda_ = np.sqrt(lambda_)
            xy = (mean[0], mean[1])
            angle=np.rad2deg(np.arctan2(*v[:,0][::-1]))
            if i == 0:
                ell = Ellipse(xy, alpha=0.3, facecolor='red',width=lambda_[0], height=lambda_[1], angle = angle, label='Confidence Ellipses')
            else:
                ell = Ellipse(xy, alpha=0.3, facecolor='red',width=lambda_[0], height=lambda_[1], angle = angle)
            ax.add_patch(ell)
            
            # draw heading
            plt.arrow(mean[0], mean[1], 
                    self.dir_length * math.cos(mean[2]), 
                    self.dir_length * math.sin(mean[2]),
                    head_width=0.02, color='black', alpha=0.5)
        plt.xlabel('X (m)')
        plt.ylabel('Y (m)')
        plt.title('EKF Estimated Trajectory')
        plt.grid(True)
        plt.axis('equal')

        x_meas = np.linspace(-0.2,0.9,12)
        y_meas = [-0.0446, 0.0010,0.03198,0.04826,0.04986,0.03678,0.0090,-0.0334,-0.09054,-0.162,-0.248,-0.35]
        
        #x_meas = np.linspace(-0.1,0.8,10)
        #y_meas = [-0.05,0.0926,0.2084,0.29738,0.359,0.39465,0.4030,0.3845,0.339,0.2669]

        #40_00_03
        #x_meas = [-0.100,0.120000,-0.80,	-0.6,	0.070,	-0.650,	-0.390,	-0.800000,	-0.0100,-0.48000,	0.0626,0.076800,0.056000000000000]
        #y_meas = [0.05,0.35,0.50,0.0300,0.500,0.7300,-0.0400,0.25,0.7200000,0.850,0.190,0.2400,	0.250000000000000]
        plt.plot(x_meas,y_meas, 'go', label='Measured State')

        plt.legend(loc='upper right')
        plt.show()


# Code to run your EKF offline with a data file.
def offline_efk():

    # Get data to filter
    filename = './data/robot_data_60_10_24_02_26_21_04_00.pkl'
    ekf_data = data_handling.get_file_data_for_kf(filename)

    # Instantiate PF with no initial guess
    x_0 = [ekf_data[0][3][0]+.5, ekf_data[0][3][1], -ekf_data[0][3][5]]
    Sigma_0 = parameters.I3
    #x_0 = [-0.2, 0.5, 3.0]      # wrong start location
    #Sigma_0 = parameters.I3    # wrong init covar
    encoder_counts_0 = ekf_data[0][2].encoder_counts
    extended_kalman_filter = ExtendedKalmanFilter(x_0, Sigma_0, encoder_counts_0)

    # Create plotting tool for ekf
    kalman_filter_plot = KalmanFilterPlot()

    # Loop over sim data
    for t in range(1, len(ekf_data)):
        row = ekf_data[t]
        delta_t = ekf_data[t][0] - ekf_data[t-1][0] # time step size
        u_t = np.array([row[2].encoder_counts, row[2].steering]) # robot_sensor_signal
        z_t = np.array([row[3][0],row[3][1],row[3][5]]) # camera_sensor_signal

        # camera y is flipped
        y_cam = row[3][1]
        y_fixed = -y_cam    
        z_t = np.array([row[3][0], y_fixed, row[3][5]])

        ##### TEMP CHANGE FOR DATA W FLIPPED ARUCO ###############
        x_cam = row[3][0]
        y_cam = row[3][1]
        theta_cam = -row[3][5]
        # rotate so 0 aligns with +x
        theta_fixed = theta_cam #+ math.pi/2
        theta_fixed = math.atan2(math.sin(theta_fixed), math.cos(theta_fixed))
        z_t = np.array([x_cam, y_fixed, theta_fixed])
        #####

        #check if robot in frame
        z_t_current = np.array([row[3][0], row[3][1], row[3][5]])
        z_t_prev = np.array([ekf_data[t-1][3][0], ekf_data[t-1][3][1], ekf_data[t-1][3][5]])
        if np.array_equal(z_t_current, z_t_prev):
            camera_sees_tag = False
        else:
            camera_sees_tag = True
        camera_sees_tag = not np.array_equal(z_t, z_t_prev) 

        all_errors = []
        if camera_sees_tag:
            error = np.linalg.norm(extended_kalman_filter.state_mean[0:2] - z_t[0:2])
            all_errors.append(error)

        # Run the EKF for a time step
        extended_kalman_filter.update(u_t, z_t, delta_t, camera_sees_tag)
        kalman_filter_plot.update(extended_kalman_filter.state_mean, extended_kalman_filter.state_covariance[0:2,0:2])
    kalman_filter_plot.save_and_plot_final()
    #kalman_filter_plot.plot_uncertainty_history()

    rmse = np.sqrt(np.mean(np.square(all_errors)))
    print(f"Final RMSE: {rmse:.4f} meters")


####### MAIN #######
if False:
    offline_efk()
