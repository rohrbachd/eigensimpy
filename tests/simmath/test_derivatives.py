import unittest
import numpy as np

from eigensimpy.simmath.Derivatives import FirstOrderForward, FirstOrderBackward, SecondOrderCentral, MixedModelDerivative


class TestFirstOrderForward(unittest.TestCase):

    def test_x_squared_forward(self):
        x = np.array([0, 1, 2, 3, 4, 5, 6, 7])
        y = x**2
        expected_dfdx = np.array([1, 3, 5, 7, 9, 11, 13, 0])

        derivative = FirstOrderForward(dim=1)
        dfdx = derivative.compute(y)

        np.testing.assert_array_almost_equal(dfdx, expected_dfdx)
        
    def test_1d(self):
        f = np.array([1, 2, 3, 4, 5])
        expected_dfdx = np.array([1, 1, 1, 1, 0])

        derivative = FirstOrderForward(dim=1)
        dfdx = derivative.compute(f)

        np.testing.assert_array_almost_equal(dfdx, expected_dfdx)

    def test_2d(self):
        f = np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ])
        expected_dfdx = np.array([
            [3, 3, 3],
            [3, 3, 3],
            [0, 0, 0]
        ])

        derivative = FirstOrderForward(dim=1)
        dfdx = derivative.compute(f)

        np.testing.assert_array_almost_equal(dfdx, expected_dfdx)

        dfdx = derivative.compute(f, 0.5)
        expected_dfdx = np.array([
                    [6, 6, 6],
                    [6, 6, 6],
                    [0, 0, 0]
                ])
        
        np.testing.assert_array_almost_equal(dfdx, expected_dfdx)
        
        derivative = FirstOrderForward(dim=2)
        dfdx = derivative.compute(f, 0.5)
        expected_dfdx = np.array([
                    [2, 2, 0],
                    [2, 2, 0],
                    [2, 2, 0]
                ])
        np.testing.assert_array_almost_equal(dfdx, expected_dfdx)
        
    def test_3d(self):
        f = np.array([
            [
                [1, 2],
                [3, 4]
            ],
            [
                [5, 6],
                [7, 8]
            ],
            [
                [9, 10],
                [11, 12]
            ]
        ])
        expected_dfdx = np.array([
            [
                [4, 4],
                [4, 4]
            ],
            [
                [4, 4],
                [4, 4]
            ],
            [
                [0, 0],
                [0, 0]
            ]
        ])

        derivative = FirstOrderForward(dim=1)
        dfdx = derivative.compute(f)

        np.testing.assert_array_almost_equal(dfdx, expected_dfdx)    
        
class TestFirstOrderBackward(unittest.TestCase):

    def test_x_squared_backward(self):
        x = np.array([0, 1, 2, 3, 4, 5, 6, 7])
        y = x**2
        expected_dfdx = np.array([0, 1, 3, 5, 7, 9, 11, 13])

        derivative = FirstOrderBackward(dim=1)
        dfdx = derivative.compute(y)

        np.testing.assert_array_almost_equal(dfdx, expected_dfdx)
        
    def test_1d(self):
        f = np.array([1, 2, 3, 4, 5])
        expected_dfdx = np.array([0, 1, 1, 1, 1])

        derivative = FirstOrderBackward(dim=1)
        dfdx = derivative.compute(f)

        np.testing.assert_array_almost_equal(dfdx, expected_dfdx)

    def test_2d(self):
        f = np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ])
        expected_dfdx = np.array([
            [0, 0, 0],
            [3, 3, 3],
            [3, 3, 3]
        ])

        derivative = FirstOrderBackward(dim=1)
        dfdx = derivative.compute(f)

        np.testing.assert_array_almost_equal(dfdx, expected_dfdx)

        dfdx = derivative.compute(f, 0.5)
        expected_dfdx = np.array([
            [0, 0, 0],
            [6, 6, 6],
            [6, 6, 6]
        ])

        np.testing.assert_array_almost_equal(dfdx, expected_dfdx)

        derivative = FirstOrderBackward(dim=2)
        dfdx = derivative.compute(f, 0.5)
        expected_dfdx = np.array([
            [0, 2, 2],
            [0, 2, 2],
            [0, 2, 2]
        ])

        np.testing.assert_array_almost_equal(dfdx, expected_dfdx)
        
    def test_3d(self):
        f = np.array([
            [
                [1, 2],
                [3, 4]
            ],
            [
                [5, 6],
                [7, 8]
            ],
            [
                [9, 10],
                [11, 12]
            ]
        ])
        expected_dfdx = np.array([
            [
                [0, 0],
                [0, 0]
            ],
            [
                [4, 4],
                [4, 4]
            ],
            [
                [4, 4],
                [4, 4]
            ]
        ])

        derivative = FirstOrderBackward(dim=1)
        dfdx = derivative.compute(f)

        np.testing.assert_array_almost_equal(dfdx, expected_dfdx)    

class TestSecondOrderCentral(unittest.TestCase):

    def test_x_squared_central(self):
        x = np.array([0, 1, 2, 3, 4, 5, 6, 7])
        y = x**2
        expected_dfdx = np.array([0, 2, 4, 6, 8, 10, 12, 0])

        derivative = SecondOrderCentral(dim=1)
        dfdx = derivative.compute(y)

        np.testing.assert_array_almost_equal(dfdx, expected_dfdx)

    def test_2d(self):
        f = np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ])
        expected_dfdx = np.array([
            [0, 1, 0],
            [0, 1, 0],
            [0, 1, 0]
        ])

        derivative = SecondOrderCentral(dim=2)
        dfdx = derivative.compute(f)

        np.testing.assert_array_almost_equal(dfdx, expected_dfdx)
        
        expected_dfdx = np.array([
            [0, 0, 0],
            [3, 3, 3],
            [0, 0, 0]
        ])

        derivative = SecondOrderCentral(dim=1)
        dfdx = derivative.compute(f)

        np.testing.assert_array_almost_equal(dfdx, expected_dfdx)


class TestMixedModelDerivative(unittest.TestCase):
    def test_derivative(self):
        x = np.arange(0, 5.1, 0.1)
        y = np.arange(0, 0.21, 0.01)

        X, Y = np.meshgrid(x, y)
        Fxy = np.cos(2 * X) - X**2 * np.exp(5 * Y) + 3 * Y**2

        # the analytical derivative of Fxy
        Fxydxdy = -10 * X * np.exp(5 * Y)

        dx = FirstOrderForward(dim=2)
        dy = FirstOrderForward(dim=1)

        deltaX = x[1] - x[0]
        deltaY = y[1] - y[0]

        dxdy = MixedModelDerivative(dy, dx)

        # numerical derivative
        ydxdy = dxdy.compute(Fxy, deltaY, deltaX)

        ydxdy[:, -1] = ydxdy[:, -2]
        ydxdy[-1, :] = ydxdy[-2, :]

        np.testing.assert_allclose(ydxdy, Fxydxdy, atol=5)
                                    
if __name__ == "__main__":
    unittest.main()