import numpy as np
import csv

class Kalman:
    def __init__(self, initial_state_covariance=0.2, measurement_covariance=5, acceleration=0, delta_t=1/20):
        self.acceleration = acceleration
        self.delta_t = delta_t

        # Transition matrix
        self.F_t = np.array([[1, 0, delta_t, 0],
                             [0, 1, 0, delta_t],
                             [0, 0, 1, 0],
                             [0, 0, 0, 1]])

        # Initial State cov
        self.P_t = np.identity(4) * initial_state_covariance

        # Process cov
        self.Q_t = np.identity(4)

        # Control matrix
        self.B_t = np.array([[0], [0], [0], [0]])

        # Control vector
        self.U_t = acceleration

        # Measurment Matrix
        self.H_t = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])

        # Measurment cov
        self.R_t = np.identity(2) * measurement_covariance

        # Initial State
        self.X_hat_t = np.array([[0], [0], [0], [0]])

    def prediction(self):
        self.X_hat_t = self.F_t.dot(self.X_hat_t) + (self.B_t.dot(self.U_t).reshape(self.B_t.shape[0], -1))
        self.P_t = np.diag(np.diag(self.F_t.dot(self.P_t).dot(self.F_t.transpose()))) + self.Q_t
        return self.X_hat_t, self.P_t

    def update(self, Z_t):
        K_prime = self.P_t.dot(self.H_t.transpose()).dot(np.linalg.inv(self.H_t.dot(self.P_t).dot(self.H_t.transpose()) + self.R_t))
        Z_t_center = Z_t[:2]
        innovation = Z_t_center - self.H_t.dot(self.X_hat_t)[:2]
        self.X_t = self.X_hat_t + K_prime.dot(innovation)
        self.P_t = self.P_t - K_prime.dot(self.H_t).dot(self.P_t)
        self.X_hat_t = self.X_t
        return self.X_t, self.P_t

    def load_car_positions(self, file_path):
        car_positions = np.genfromtxt(file_path, delimiter=',', skip_header=1)
        return car_positions

    def run_kalman_filter(self, input_file, output_file):
        car_positions = self.load_car_positions(input_file)
        with open(output_file, mode='w', newline='') as output_file:
            output_writer = csv.writer(output_file)
            output_writer.writerow(['Frame', 'Prediction_X', 'Prediction_Y'])

            for i in range(car_positions.shape[0]):
                frame_num = int(car_positions[i, 0])
                X_hat_t, P_hat_t = self.prediction()
                X_hat_xy = X_hat_t[:2, 0]

                Z_t = car_positions[i, 2:4].transpose()
                Z_t = Z_t.reshape(Z_t.shape[0], -1)

                X_t, P_t = self.update(Z_t)
                output_writer.writerow([frame_num] + X_t[:2].flatten().tolist())

# Example usage
# kalman_filter = Kalman()
# kalman_filter.run_kalman_filter('2-ODmodel/object_positions.csv', '2-ODmodel/kalman_output.csv')
