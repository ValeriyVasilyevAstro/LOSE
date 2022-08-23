import numpy as np


def createSeriesOf15ImagesUsingQuadraticPolynom() -> np.ndarray:
    n_t = 15
    nx = 3
    ny = 3
    time = np.linspace(0, 10, n_t)
    flux_single_pixel = time**3 - 0.5*time**2 + 10*time + 4
    images = np.zeros((n_t, nx, ny))
    for i in range(nx):
        for j in range(ny):
            images[:, i, j] = flux_single_pixel
    return images