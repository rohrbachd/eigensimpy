import unittest
import numpy as np

# Import square function
from eigensimpy.dsp.Signals import Quantity, Dimension, DimensionArray

#elp(Quantity)

class TestQuantity(unittest.TestCase):
    def test_label_with_si_unit(self):
        q = Quantity(name='Length', si_unit='m')
        self.assertEqual(q.label, 'Length [m]')

    def test_label_without_si_unit(self):
        q = Quantity(name='Time')
        self.assertEqual(q.label, 'Time')

    def test_equality(self):
        q1 = Quantity(name='Length', si_unit='m')
        q2 = Quantity(name='Length', si_unit='m')
        self.assertEqual(q1, q2)

    def test_inequality(self):
        q1 = Quantity(name='Length', si_unit='m')
        q2 = Quantity(name='Time', si_unit='s')
        self.assertNotEqual(q1, q2)

class TestDimension(unittest.TestCase):
    def setUp(self):
        self.dim1 = Dimension(delta=2,   offset=1, quantity=Quantity(name='Time',       si_unit='s'))
        self.dim2 = Dimension(delta=0.1, offset=0, quantity=Quantity(name='Frequency',  si_unit='Hz'))
        self.dim3 = Dimension(delta=1,   offset=0, quantity=Quantity(name='Length',     si_unit='m'))
       
    def test_si_unit(self):
        self.assertEqual(self.dim1.si_unit, 's')
        self.assertEqual(self.dim2.si_unit, 'Hz')
        self.assertEqual(self.dim3.si_unit, 'm')

    def test_name(self):
        self.assertEqual(self.dim1.name, 'Time')
        self.assertEqual(self.dim2.name, 'Frequency')
        self.assertEqual(self.dim3.name, 'Length')

    def test_label(self):
        self.assertEqual(self.dim1.label, 'Time [s]')
        self.assertEqual(self.dim2.label, 'Frequency [Hz]')
        self.assertEqual(self.dim3.label, 'Length [m]')

    def test_eq(self):
        dim4 = Dimension(delta=2, offset=1, quantity=Quantity(name='Time', si_unit='s'))
        self.assertEqual(self.dim1, dim4)
        self.assertNotEqual(self.dim1, self.dim2)

    def test_dimVector(self):
        vec1 = np.array([1, 3, 5, 7])
        vec2 = np.array([0, 0.1, 0.2, 0.1*3])
        vec3 = np.array([0, 1, 2, 3])
        self.assertTrue(np.array_equal(self.dim1.dimVector(4), vec1))
        self.assertTrue(np.array_equal(self.dim2.dimVector(4), vec2))
        self.assertTrue(np.array_equal(self.dim3.dimVector(4), vec3))
        
        
class TestDimensionArray(unittest.TestCase):
    def setUp(self):
        self.dim_array = np.array([
            Dimension(delta=1.0, offset=0.0, quantity=Quantity(si_unit='m', name='length', label='L')),
            Dimension(delta=2.0, offset=1.0, quantity=Quantity(si_unit='s', name='time', label='T')),
            Dimension(delta=0.5, offset=2.0, quantity=Quantity(si_unit='kg', name='mass', label='M'))
        ])
        self.da = DimensionArray(self.dim_array)
    
    def test_find_dim_name(self):
        self.assertTrue(np.array_equal(self.da.find_dim_name('length'), [True, False, False]))
        self.assertTrue(np.array_equal(self.da.find_dim_name('time'), [False, True, False]))
        self.assertTrue(np.array_equal(self.da.find_dim_name('mass'), [False, False, True]))
        self.assertTrue(np.array_equal(self.da.find_dim_name('energy'), [False, False, False]))
        
    def test_find_dim_nameIndex(self):
        self.assertTrue(np.array_equal(self.da.find_dim_nameIndex('length'), [0]))
        self.assertTrue(np.array_equal(self.da.find_dim_nameIndex('energy'), []))
        
    def test_set_dim_unit(self):
        self.da.set_dim_unit(0, 'cm')
        self.assertEqual(self.da.dim_array[0].quantity.si_unit, 'cm')
        
        self.da.set_dim_unit([1, 2], ['min', 'lb'])
        self.assertEqual(self.da.dim_array[1].quantity.si_unit, 'min')
        self.assertEqual(self.da.dim_array[2].quantity.si_unit, 'lb')
    
    def test_set_dim_name(self):
        self.da.set_dim_name(0, 'distance')
        self.assertEqual(self.da.dim_array[0].quantity.name, 'distance')
        
        self.da.set_dim_name([1, 2], ['duration', 'weight'])
        self.assertEqual(self.da.dim_array[1].quantity.name, 'duration')
        self.assertEqual(self.da.dim_array[2].quantity.name, 'weight')
    
    def test_set_dim_delta(self):
        self.da.set_dim_delta(0, 0.1)
        self.assertEqual(self.da.dim_array[0].delta, 0.1)
        
        self.da.set_dim_delta([1, 2], [0.2, 0.3])
        self.assertEqual(self.da.dim_array[1].delta, 0.2)
        self.assertEqual(self.da.dim_array[2].delta, 0.3)
    
    def test_set_dim_offset(self):
        self.da.set_dim_offset(0, 0.5)
        self.assertEqual(self.da.dim_array[0].offset, 0.5)
        
        self.da.set_dim_offset([1, 2], [1.5, 2.5])
        self.assertEqual(self.da.dim_array[1].offset, 1.5)
        self.assertEqual(self.da.dim_array[2].offset, 2.5)
    
    def test_input_array_error(self):
        with self.assertRaises(ValueError):
            DimensionArray([1, 2, 3])
            
        with self.assertRaises(ValueError):
            DimensionArray([Dimension(), 'not a dimension', Dimension()])
            
if __name__ == '__main__':
    unittest.main()