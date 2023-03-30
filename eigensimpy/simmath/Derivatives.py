import numpy as np


class FirstOrderForward:
    """
    Simple first order derivative
    """

    def __init__(self, dim=1):
        self.dimension = dim

    def compute(self, f, dx=1):
        if self.dimension > 4:
            raise ValueError("Given dimension is not supported")

        dfdx = np.zeros(f.shape)
        dd = np.diff(f, axis=self.dimension - 1) / dx

        # assign all elements of dd to dfdx except the last one 
        # in the first dimension of the array.
        if self.dimension == 1:
            dfdx[:-1, ...] = dd  
        elif self.dimension == 2:
            dfdx[:, :-1, ...] = dd
        elif self.dimension == 3:
            dfdx[:, :, :-1, ...] = dd
        elif self.dimension == 4:
            dfdx[:, :, :, :-1] = dd

        return dfdx
    
class FirstOrderBackward:
    """
    Simple first order backward derivative
    """

    def __init__(self, dim=1):
        self.dimension = dim

    def compute(self, f, dx=1):
        if self.dimension > 4:
            raise ValueError("Given dimension is not supported")

        dfdx = np.zeros(f.shape)
        dd = np.diff(f, axis=self.dimension - 1) / dx

        # assign all elements of dd to dfdx except the first one 
        # in the first dimension of the array.
        if self.dimension == 1:
            dfdx[1:, ...] = dd
        elif self.dimension == 2:
            dfdx[:, 1:, ...] = dd
        elif self.dimension == 3:
            dfdx[:, :, 1:, ...] = dd
        elif self.dimension == 4:
            dfdx[:, :, :, 1:] = dd

        return dfdx
    
class SecondOrderCentral:
    def __init__(self, dim):
        self.Dimension = dim

    def compute(self, f, dx=1):
        dim = self.Dimension

        dfdx = np.zeros_like(f)

        dd = (np.roll(f, -1, axis=dim - 1) - np.roll(f, 1, axis=dim - 1)) / (2 * dx)

        if dim == 1:
            dfdx[1:-1, ...] = dd[1:-1, ...]
        elif dim == 2:
            dfdx[:, 1:-1, ...] = dd[:, 1:-1, ...]
        elif dim == 3:
            dfdx[:, :, 1:-1, ...] = dd[:, :, 1:-1, ...]
        elif dim == 4:
            dfdx[:, :, :, 1:-1] = dd[:, :, :, 1:-1]
        else:
            raise ValueError("Given dimension is not supported")

        return dfdx    
    
class MixedModelDerivative:
    def __init__(self, dim1_derivative, dim2_derivative):
        self.dim1_derivative = dim1_derivative
        self.dim2_derivative = dim2_derivative

    def compute(self, f, dx1=1, dx2=1):
        mixed_dfdxdy = self.dim1_derivative.compute(f, dx1)
        mixed_dfdxdy = self.dim2_derivative.compute(mixed_dfdxdy, dx2)

        return mixed_dfdxdy