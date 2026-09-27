import numpy as np


def flops(image_size, batch):
    s, b = np.asarray(image_size, float), np.asarray(batch, float)
    return b * (17714 * s**2 + 313700)


def memory(image_size, batch):
    s, b = np.asarray(image_size, float), np.asarray(batch, float)
    return 4161296 + 68 * b * s**2


def latency(image_size, batch, theta):
    s, b = np.asarray(image_size, float), np.asarray(batch, float)
    moved = 4 * (95 * b * s**2 + 2148 * b + 1040324)
    return theta['overhead_s'] + np.maximum(
        flops(image_size, batch) / theta['flops_per_s'],
        moved / theta['bytes_per_s'])


def energy(image_size, batch, theta_energy):
    return theta_energy['power_w'] * latency(
        image_size, batch, theta_energy['latency_theta'])
