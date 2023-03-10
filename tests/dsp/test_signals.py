import unittest
import numpy as np

# Import square function
from eigensimpy.dsp.Signals import Quantity, Dimension

help(Quantity)

class TestQuantity(unittest.TestCase):
    def test_label_with_si_unit(self):
        q = Quantity(Name='Length', SiUnit='m')
        self.assertEqual(q.Label, 'Length [m]')

    def test_label_without_si_unit(self):
        q = Quantity(Name='Time')
        self.assertEqual(q.Label, 'Time')

    def test_equality(self):
        q1 = Quantity(Name='Length', SiUnit='m')
        q2 = Quantity(Name='Length', SiUnit='m')
        self.assertEqual(q1, q2)

    def test_inequality(self):
        q1 = Quantity(Name='Length', SiUnit='m')
        q2 = Quantity(Name='Time', SiUnit='s')
        self.assertNotEqual(q1, q2)

class TestDimension(unittest.TestCase):
    def setUp(self):
        self.dim1 = Dimension(Delta=2, Offset=1, Quantity=Quantity(Name='Time', SiUnit='s'))
        self.dim2 = Dimension(Delta=0.1, Offset=0, Quantity=Quantity(Name='Frequency', SiUnit='Hz'))
        self.dim3 = Dimension(Delta=1, Offset=0, Quantity=Quantity(Name='Length', SiUnit='m'))

    def test_SiUnit(self):
        self.assertEqual(self.dim1.SiUnit, 's')
        self.assertEqual(self.dim2.SiUnit, 'Hz')
        self.assertEqual(self.dim3.SiUnit, 'm')

    def test_Name(self):
        self.assertEqual(self.dim1.Name, 'Time')
        self.assertEqual(self.dim2.Name, 'Frequency')
        self.assertEqual(self.dim3.Name, 'Length')

    def test_Label(self):
        self.assertEqual(self.dim1.Label, 'Time [s]')
        self.assertEqual(self.dim2.Label, 'Frequency [Hz]')
        self.assertEqual(self.dim3.Label, 'Length [m]')

    def test_eq(self):
        dim4 = Dimension(Delta=2, Offset=1, Quantity=Quantity(Name='Time', SiUnit='s'))
        self.assertEqual(self.dim1, dim4)
        self.assertNotEqual(self.dim1, self.dim2)

    def test_dimVector(self):
        vec1 = np.array([1, 3, 5, 7])
        vec2 = np.array([0, 0.1, 0.2, 0.3])
        vec3 = np.array([0, 1, 2, 3])
        self.assertTrue(np.array_equal(self.dim1.dimVector(4), vec1))
        self.assertTrue(np.array_equal(self.dim2.dimVector(4), vec2))
        self.assertTrue(np.array_equal(self.dim3.dimVector(4), vec3))

    def test_findDimName(self):
        self.assertTrue(np.array_equal(self.dim1.findDimName('Time'), np.array([True])))
        self.assertTrue(np.array_equal(self.dim2.findDimName('Frequency'), np.array([True])))
        self.assertTrue(np.array_equal(self.dim3.findDimName('Length'), np.array([True])))

    def test_findDimNameIndex(self):
        self.assertTrue(np.array_equal(self.dim1.findDimNameIndex('Time'), np.array([0])))
        self.assertTrue(np.array_equal(self.dim2.findDimNameIndex('Frequency'), np.array([1])))
        self.assertTrue(np.array_equal(self.dim3.findDimNameIndex('Length'), np.array([2])))

    def test_setDimUnit(self):
        self.dim1.setDimUnit(0, 'ms')
        self.assertEqual(self.dim1.SiUnit, 'ms')
        self.dim2.setDimUnit(slice(None), 'kHz')
        self.assertTrue(np.array_equal([d.SiUnit for d in [self.dim1, self.dim2, self.dim3]], ['ms', 'kHz', 'm']))
        with self.assertRaises(ValueError):
            self.dim3.setDimUnit([0, 1], 'm/s')
            
if __name__ == '__main__':
    unittest.main()