"""Simple Collection of functions to generate 2d receptive field filters."""


import numpy as np


__all__ = [
    "gaussian",
    "ricker_wavelet",
    "gabor",
]


def gaussian(size=16, center=(8, 8), amplitude=1.0, sigma=1.0):
    """Generate a 2D Gaussian filter.

    Note:
        The boarders are not cyclic, so the Gaussian will be truncated at the edges.
        This is for our usecase an expected behavior.

    Parameters:
    - size (int): Size of the square grid (default is 16).
    - center (tuple): (x0, y0) coordinates of the Gaussian center.
    - amplitude (float): Peak value of the Gaussian.
    - sigma (float): Standard deviation of the Gaussian.

    Returns:
    - 2D NumPy array representing the Gaussian distribution.
    """
    x = np.arange(0, size, 1)
    y = np.arange(0, size, 1)
    x, y = np.meshgrid(x, y)

    x0, y0 = center
    gaussian = amplitude * np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))
    return gaussian


def ricker_wavelet(size=16, center=(8, 8), amplitude=1.0, sigma=2.0):
    """
    Generate a 2D Mexican Hat (Laplacian of Gaussian) filter.

    Note:
        The boarders are not cyclic, so the filter will be truncated at the edges.
        This is for our usecase an expected behavior.

    Parameters:
    - size (int): Size of the square grid (default is 16).
    - center (tuple): (x0, y0) coordinates of the filter center.
    - amplitude (float): Peak amplitude of the filter.
    - sigma (float): Standard deviation of the Gaussian.

    Returns:
    - 2D NumPy array representing the Mexican Hat filter.
    """
    x = np.arange(0, size, 1)
    y = np.arange(0, size, 1)
    x, y = np.meshgrid(x, y)

    x0, y0 = center
    r_squared = (x - x0)**2 + (y - y0)**2
    factor = (1 - (1/2) * (r_squared / sigma**2)) / (np.pi*sigma**4)
    gaussian = np.exp(-r_squared / (2 * sigma**2))
    mexican_hat = amplitude * factor * gaussian
    return mexican_hat


def gabor(size=16, center=(8, 8), amplitude=1.0, sigma=2.0,
                      theta=0, frequency=0.25, phase=0):
    """
    Generate a 2D Gabor filter.

    Note:
        The boarders are not cyclic, so the filter will be truncated at the edges.
        This is for our usecase an expected behavior.

    Parameters:
    - size (int): Size of the square grid (default is 16).
    - center (tuple): (x0, y0) coordinates of the Gabor center.
    - amplitude (float): Peak value of the Gabor function.
    - sigma (float): Standard deviation of the Gaussian envelope.
    - theta (float): Orientation of the Gabor filter in radians.
    - frequency (float): Spatial frequency of the sinusoidal component.
    - phase (float): Phase offset of the sinusoidal component in radians.

    Returns:
    - 2D NumPy array representing the Gabor filter.
    """
    x = np.arange(0, size, 1)
    y = np.arange(0, size, 1)
    x, y = np.meshgrid(x, y)

    x0, y0 = center
    x_shifted = x - x0
    y_shifted = y - y0

    # Rotation
    x_theta = x_shifted * np.cos(theta) + y_shifted * np.sin(theta)
    y_theta = -x_shifted * np.sin(theta) + y_shifted * np.cos(theta)

    # Gabor formula
    gaussian_envelope = np.exp(-0.5 * (x_theta**2 + y_theta**2) / sigma**2)
    sinusoidal_component = np.cos(2 * np.pi * frequency * x_theta + phase)
    gabor = amplitude * gaussian_envelope * sinusoidal_component

    return gabor

