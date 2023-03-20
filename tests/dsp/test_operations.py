import unittest
import numpy as np
import eigensimpy.dsp.Operations as ops

from eigensimpy.dsp.Signals import Signal, Dimension

class TestSignalOperations(unittest.TestCase):

    def setUp(self):
        self.signal1 = Signal(data=np.array([1, 4, 9]), dims=Dimension(name='Time', si_unit='s'))
        self.signal2 = Signal(data=np.array([0, np.pi / 2, np.pi]), dims=Dimension(name='Angle', si_unit='rad'))
        self.signal3 = Signal(data=np.array([0, -1, -9]), dims=Dimension(name='Time', si_unit='s'))
        self.signal4 = Signal(data=np.array([0, 0.5]), dims=Dimension(name='Time', si_unit='s'))
         
    def test_sqrt(self):
        result = ops.sqrt(self.signal1)
        expected_data = np.array([1, 2, 3])
        np.testing.assert_array_equal(result.data, expected_data)
        self.assertEqual(result.dims, self.signal1.dims)
        self.assertEqual(result.amplitude, self.signal1.amplitude)
        
        # test np option
        result = ops.sqrt(self.signal1.data)
        np.testing.assert_array_equal(result.data, expected_data)
        

    def test_cos(self):
        result = ops.cos(self.signal2)
        expected_data = np.array([1, 0, -1])
        np.testing.assert_array_almost_equal(result.data, expected_data, decimal=8)
        self.assertEqual(result.dims, self.signal2.dims)
        self.assertEqual(result.amplitude, self.signal2.amplitude)

        # test np option
        result = ops.cos(self.signal2.data)
        np.testing.assert_array_almost_equal(result, expected_data, decimal=8)
        
        
    def test_sin(self):
        result = ops.sin(self.signal2)
        expected_data = np.array([0, 1, 0])
        np.testing.assert_array_almost_equal(result.data, expected_data, decimal=8)
        self.assertEqual(result.dims, self.signal2.dims)
        self.assertEqual(result.amplitude, self.signal2.amplitude)
        
        # test np option
        result = ops.sin(self.signal2.data)
        np.testing.assert_array_almost_equal(result, expected_data, decimal=8)

    def test_log(self):
        result = ops.log(self.signal1)
        expected_data = np.array([0, np.log(4), np.log(9)])
        np.testing.assert_array_almost_equal(result.data, expected_data, decimal=8)
        self.assertEqual(result.dims, self.signal1.dims)
        self.assertEqual(result.amplitude, self.signal1.amplitude)
        
        # test np option
        result = ops.log(self.signal1.data)
        np.testing.assert_array_almost_equal(result, expected_data, decimal=8)

    def test_log10(self):
        result = ops.log10(self.signal1)
        expected_data = np.array([0, np.log10(4), np.log10(9)])
        np.testing.assert_array_almost_equal(result.data, expected_data, decimal=8)
        self.assertEqual(result.dims, self.signal1.dims)
        self.assertEqual(result.amplitude, self.signal1.amplitude)
        
        # test np option
        result = ops.log10(self.signal1.data)
        np.testing.assert_array_almost_equal(result, expected_data, decimal=8)

    def test_arcsin(self):        
        result = ops.arcsin(self.signal4)
        expected_data = np.array([0, np.arcsin(0.5)])
        np.testing.assert_array_almost_equal(result.data, expected_data, decimal=8)
        self.assertEqual(result.dims, self.signal4.dims)
        self.assertEqual(result.amplitude, self.signal4.amplitude)
        
        # test np option
        result = ops.arcsin(self.signal4.data)
        np.testing.assert_array_almost_equal(result, expected_data, decimal=8)


    def test_arccos(self):
        result = ops.arccos(self.signal4)
        expected_data = np.array([np.arccos(0), np.arccos(0.5)])
        np.testing.assert_array_almost_equal(result.data, expected_data, decimal=8)
        self.assertEqual(result.dims, self.signal4.dims)
        self.assertEqual(result.amplitude, self.signal4.amplitude)
        
        # test np option
        result = ops.arccos(self.signal4.data)
        np.testing.assert_array_almost_equal(result, expected_data, decimal=8)


    def test_tan(self):
        result = ops.tan(self.signal3)
        expected_data = np.array([np.tan(0), np.tan(-1), np.tan(-9)])
        np.testing.assert_array_almost_equal(result.data, expected_data, decimal=8)
        self.assertEqual(result.dims, self.signal3.dims)
        self.assertEqual(result.amplitude, self.signal3.amplitude)
        
        # test np option
        result = ops.tan(self.signal3.data)
        np.testing.assert_array_almost_equal(result, expected_data, decimal=8)


    def test_arctan(self):
        result = ops.arctan(self.signal3)
        expected_data = np.array([np.arctan(0), np.arctan(-1), np.arctan(-9)])
        np.testing.assert_array_almost_equal(result.data, expected_data, decimal=8)
        self.assertEqual(result.dims, self.signal3.dims)
        self.assertEqual(result.amplitude, self.signal3.amplitude)
        
        # test np option
        result = ops.arctan(self.signal3.data)
        np.testing.assert_array_almost_equal(result, expected_data, decimal=8)

if __name__ == '__main__':
    unittest.main()