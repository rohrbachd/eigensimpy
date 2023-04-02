
import numpy as np
from numpy.testing import assert_array_almost_equal
import unittest

from eigensimpy.simmath.MathUtil import interp2, to_vec_coords

class TestMathUtil(unittest.TestCase):
    
    def test_interp2(self):
        z = np.array([[1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9]])

        xq, yq = np.meshgrid(np.linspace(1, 3, 5), np.linspace(1, 3, 5))

        expected = np.array([[1.  , 1.5 , 2.  , 2.5 , 3.  ],
                            [2.5,   3,    3.5,  4,    4.5 ],
                            [4.  ,  4.5 , 5.  , 5.5 , 6.  ],
                            [5.5,   6,    6.5,  7.,   7.5 ],
                            [7.  ,  7.5 , 8.  , 8.5 , 9.  ]])

        result = interp2(z, xq, yq)
        assert_array_almost_equal(result, expected, decimal=2)
        
        
class TestToVecCoords(unittest.TestCase):
    
    def test_2D_case(self):
        x = [1, 2, 3]
        y = [4, 5, 6]
        X, Y = to_vec_coords(x, y)
        self.assertIsInstance(X, np.ndarray)
        self.assertIsInstance(Y, np.ndarray)
        self.assertEqual(X.shape, (9,))
        self.assertEqual(Y.shape, (9,))
        self.assertTrue(np.allclose(X, [1, 1, 1, 2, 2, 2, 3, 3, 3]))
        self.assertTrue(np.allclose(Y, [4, 5, 6, 4, 5, 6, 4, 5, 6]))

    def test_3D_case(self):
        
        x = [1, 2]
        y = [3, 4, 5]
        z = [6, 7, 8, 9]
        X, Y, Z = to_vec_coords(x, y, z)
        self.assertIsInstance(X, np.ndarray)
        self.assertIsInstance(Y, np.ndarray)
        self.assertIsInstance(Z, np.ndarray)
        self.assertEqual(X.shape, (24,))
        self.assertEqual(Y.shape, (24,))
        self.assertEqual(Z.shape, (24,))
        self.assertTrue(np.allclose(X, [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]))
        self.assertTrue(np.allclose(Y, [3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5]))
        self.assertTrue(np.allclose(Z, [6, 7, 8, 9, 6, 7, 8, 9, 6, 7, 8, 9, 6, 7, 8, 9, 6, 7, 8, 9, 6, 7, 8, 9]))    
        
        
if __name__ == '__main__':
    unittest.main()    