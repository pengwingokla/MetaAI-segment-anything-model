import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt
from scipy.stats import norm
from kalman import kf_internal
from collections import namedtuple

class KalmanFilter:
    gaussian = namedtuple('Gaussian', ['mean', 'var'])

    def __init__(self, input_csv, fname_out='kalman-distribution.png', kalman_csv='kalman_results.csv'):
        self.INPUT_CSV = input_csv
        self.FNAME_OUT = fname_out
        self.KALMAN_CSV = kalman_csv
        self.process_var = 0
        self.sensor_var = 0
        self.x = None
        self.y = None
        self.mu_x = 0
        self.std_x = 0
        self.mu_y = 0
        self.std_y = 0
        self.center_x = None
        self.center_y = None

    def fit_gaussians(self):
        data = pd.read_csv(self.INPUT_CSV)
        self.center_x = data['Center X']
        self.center_y = data['Center Y']
        self.mu_x, self.std_x = norm.fit(self.center_x)
        self.mu_y, self.std_y = norm.fit(self.center_y)

    def plot_histograms(self):
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.hist(self.center_x, bins=20, density=True, alpha=0.6, color='blue')
        x_range = np.linspace(min(self.center_x), max(self.center_x), 100)
        plt.plot(x_range, norm.pdf(x_range, self.mu_x, self.std_x), label='Gaussian Fit', color='black')
        plt.xlabel('Center X')
        plt.ylabel('Density')
        plt.title('Belief of Car\'s Position (X)')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.hist(self.center_y, bins=20, density=True, alpha=0.6, color='blue', orientation='horizontal')
        y_range = np.linspace(min(self.center_y), max(self.center_y), 100)
        plt.plot(norm.pdf(y_range, self.mu_y, self.std_y), y_range, label='Gaussian Fit', color='black')
        plt.xlabel('Density')
        plt.ylabel('Center Y')
        plt.title('Belief of Car\'s Position (Y)')
        plt.legend()

        plt.tight_layout()
        plt.savefig(self.FNAME_OUT)

    def kalman_filter(self):
        self.process_var = self.std_x**2.
        self.sensor_var = self.std_x**2.
        gaussian = self.gaussian
        gaussian.__repr__ = lambda s: f'𝒩(μ={s[0]:.3f}, 𝜎²={s[1]:.3f})'
        self.x = gaussian(self.mu_x, self.std_x)
        self.y = gaussian(self.mu_y, self.std_y)

        velocity = 1
        dt = 1.
        process_model = gaussian(velocity*dt, self.process_var)

        zxs = self.center_x.tolist()
        zys = self.center_y.tolist()

        prior_x = []
        updated_x = []
        prior_y = []
        updated_y = []

        print('PREDICT\t\t\tUPDATE')
        print('     x      var\t\t  z\t    x      var')

        for z in zxs:    
            prior = self.predict(self.x, process_model)
            likelihood = gaussian(z, self.sensor_var)
            prior_x.append(prior[0])
            updated_x.append(self.x[0])
            kf_internal.print_gh(prior, self.x, z)

        for z in zys:    
            prior = self.predict(self.y, process_model)
            likelihood = gaussian(z, self.sensor_var)
            prior_y.append(prior[0])
            updated_y.append(self.y[0])
            kf_internal.print_gh(prior, self.y, z)

        print()
        print(f'final estimate:        {self.x.mean:10.3f}')
        print(f'actual final position: {self.mu_x:10.3f}')

        # Write the updated x_pos list to a CSV file
        with open(self.KALMAN_CSV, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Prior_x','Prior_y', 'Updated_x', 'Updated_y'])  # Write header
            for px, py, ux, uy in zip(prior_x, prior_y, updated_x, updated_y):
                writer.writerow([px, py, ux, uy])

        print("Updated values of x.mean have been saved to 'kalman_results.csv' file.")

    def predict(self, pos, movement):
        return self.gaussian(pos.mean + movement.mean, pos.var + movement.var)

    def gaussian_multiply(self, g1, g2):
        return kf_internal.gaussian_multiply(g1, g2)

    def update(self, prior, likelihood):
        return kf_internal.update(prior, likelihood)

# Example usage
if __name__ == "__main__":

    INPUT_CSV = 'assignment-3-drone-kalman-filter\\3-Kalman\detected_pos_vid1.csv'
    FNAME_OUT = 'kalman-distribution.png'
    KALMAN_CSV= 'kalman_results.csv'


    kf = KalmanFilter(INPUT_CSV, FNAME_OUT, KALMAN_CSV)
    kf.fit_gaussians()
    kf.plot_histograms()
    kf.kalman_filter()
