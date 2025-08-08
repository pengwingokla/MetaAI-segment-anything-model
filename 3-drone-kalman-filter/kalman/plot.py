import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt
from scipy.stats import norm
import kf_internal
from collections import namedtuple
import book_plots as book_plots
from ipywidgets.widgets import IntSlider
from ipywidgets import interact


INPUT_CSV = 'assignment-3-drone-kalman-filter\\3-Kalman\detected_pos_vid1.csv'
FNAME_OUT = 'kalman-distribution.png'
KALMAN_CSV= 'kalman_results.csv'

data = pd.read_csv(INPUT_CSV)

# Extract center x and center y coordinates
center_x = data['Center X']
center_y = data['Center Y']

# Fit Gaussian distributions to the data
mean_x, stdev_x = norm.fit(center_x)
mean_y, stdev_y = norm.fit(center_y)

print("Center X - Mean:", mean_x, "Standard Deviation:", stdev_x)
print("Center Y - Mean:", mean_y, "Standard Deviation:", stdev_y)

# Load the CSV file containing center x and center y coordinates
data = pd.read_csv(INPUT_CSV)

# Exclude the first row
data = data.iloc[3:]

# Extract center x and center y coordinates
center_x = data['Center X']
center_y = data['Center Y']

# Fit Gaussian distributions to the data
mu_x, std_x = norm.fit(center_x)
mu_y, std_y = norm.fit(center_y)

# Plot center x
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.hist(center_x, bins=20, density=True, alpha=0.6, color='blue')
x_range = np.linspace(min(center_x), max(center_x), 100)
plt.plot(x_range, norm.pdf(x_range, mu_x, std_x), label='Gaussian Fit', color='black')
plt.xlabel('Center X')
plt.ylabel('Density')
plt.title('Belief of Car\'s Position (X)')
plt.legend()

# Plot center y
plt.subplot(1, 2, 2)
plt.hist(center_y, bins=20, density=True, alpha=0.6, color='blue', orientation='horizontal')
y_range = np.linspace(min(center_y), max(center_y), 100)
plt.plot(norm.pdf(y_range, mu_y, std_y), y_range, label='Gaussian Fit', color='black')
plt.xlabel('Density')
plt.ylabel('Center Y')
plt.title('Belief of Car\'s Position (Y)')
plt.legend()

plt.tight_layout()
# plt.show()

# plt.savefig(FNAME_OUT)

process_var = stdev_x**2.   # variance in the car's movement
sensor_var = std_x**2.      # variance in the sensor

gaussian = namedtuple('Gaussian', ['mean', 'var'])
gaussian.__repr__ = lambda s: f'𝒩(μ={s[0]:.3f}, 𝜎²={s[1]:.3f})'

x = gaussian(mean_x, stdev_x)  # car's position, N(mean, stdev)
y = gaussian(mean_y, stdev_y)  # car's position, N(mean, stdev)

velocity = 1
dt = 1. # time step in seconds
process_model = gaussian(velocity*dt, process_var) # displacement to add to x

def predict(pos, movement):
    return gaussian(pos.mean + movement.mean, pos.var + movement.var)
def gaussian_multiply(g1, g2):
    mean = (g1.var * g2.mean + g2.var * g1.mean) / (g1.var + g2.var)
    variance = (g1.var * g2.var) / (g1.var + g2.var)
    return gaussian(mean, variance)
def update(prior, likelihood):
    posterior = gaussian_multiply(likelihood, prior)
    return posterior

zxs = center_x.tolist()
zys = center_y.tolist()

prior_x = []
updated_x = []
print('PREDICT\t\t\tUPDATE')
print('     x      var\t\t  z\t    x      var')

# perform Kalman filter on measurement z
for z in zxs:    
    prior = predict(x, process_model)
    likelihood = gaussian(z, sensor_var)
    prior_x.append(prior[0])
    updated_x.append(x[0])
    kf_internal.print_gh(prior, x, z)

prior_y = []
updated_y = []

for z in zxs:    
    prior = predict(y, process_model)
    likelihood = gaussian(z, sensor_var)
    prior_y.append(prior[0])
    updated_y.append(y[0])
    kf_internal.print_gh(prior, y, z)

print()
print(f'final estimate:        {x.mean:10.3f}')
print(f'actual final position: {mu_x:10.3f}')

# Write the updated x_pos list to a CSV file
with open(KALMAN_CSV, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Prior_x','Prior_y', 'Updated_x', 'Updated_y'])  # Write header
    for px, py, ux, uy in zip(prior_x, prior_y, updated_x, updated_y):
        writer.writerow([px, py, ux, uy])

print("Updated values of x.mean have been saved to 'kalman_results.csv' file.")
