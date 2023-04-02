

from scipy.interpolate import griddata

import numpy as np


def interp2(z, xq, yq, method='linear'):
    x, y = np.meshgrid(np.arange(1, z.shape[1] + 1), np.arange(1, z.shape[0] + 1))

    points = np.column_stack((x.ravel(), y.ravel()))
    values = z.ravel()

    grid_points = np.column_stack((xq.ravel(), yq.ravel()))

    interpolated_values = griddata(points, values, grid_points, method=method)
    return interpolated_values.reshape(xq.shape)


def to_vec_coords(x, y, z=None):
    if z is None:
        X, Y = np.meshgrid(x, y, indexing='ij')
        X = X.flatten()
        Y = Y.flatten()
        return X, Y
    else:
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        X = X.flatten()
        Y = Y.flatten()
        Z = Z.flatten()
        return X, Y, Z