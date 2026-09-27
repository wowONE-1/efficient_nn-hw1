import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

rows = pd.read_csv('results/measurements.csv')
train = rows[
    (rows.status == 'ok')
    & rows.image_size.isin([32, 64, 128, 224, 256, 384, 512])
    & rows.batch.isin([1, 2, 4, 8, 16, 32, 64, 128, 256])
]
s = train.image_size.to_numpy()
b = train.batch.to_numpy()
f = b * (17714 * s**2 + 313700)
d = 4 * (95 * b * s**2 + 2148 * b + 1040324)
t = train.latency_s.to_numpy()

fit = least_squares(
    lambda x: np.log((10**x[0] + np.maximum(f / 10**x[1], d / 10**x[2])) / t),
    [-3, 12, 10], bounds=([-9, 7, 5], [0, 16, 15]))
theta = dict(zip(['overhead_s', 'flops_per_s', 'bytes_per_s'], 10**fit.x))

energy = None
valid = train.energy_j > 0
if valid.any():
    predicted_time = (theta['overhead_s'] + np.maximum(
        f[valid] / theta['flops_per_s'], d[valid] / theta['bytes_per_s']))
    measured_energy = train.loc[valid, 'energy_j'].to_numpy()
    power = np.dot(predicted_time, measured_energy) / np.dot(predicted_time, predicted_time)
    energy = {'power_w': float(power), 'latency_theta': theta}

Path('results/theta.json').write_text(json.dumps({'latency': theta, 'energy': energy}, indent=2))
print('Parameters saved to results/theta.json')
