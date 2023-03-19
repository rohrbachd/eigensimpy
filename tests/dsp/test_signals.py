import unittest
import numpy as np

# Import square function
from eigensimpy.dsp.Signals import Signal, Quantity, Dimension, DimensionArray
import matplotlib.pyplot as plt
#elp(Quantity)

class SignalTests(unittest.TestCase):
    
    def setUp(self):
        # Test cropping along the first dimension
        data = np.array([[1,  2,  3,  4 ],
                         [4,  5,  6,  7], 
                         [7,  8,  9,  10], 
                         [10, 11, 12, 13],
                         [1,  2,  3,  4 ],
                         [4,  5,  6,  7],
                         [7,  8,  9,  10],])
        
        dims = DimensionArray([Dimension(quantity=Quantity(name="time", si_unit="s"), delta = 0.5, offset = 2),
                               Dimension(quantity=Quantity(name="channel", si_unit="V"), delta = 2.0, offset = 3.1)])
        amplitude = Quantity(name='Amplitude', si_unit='')
        self.signal = Signal(data=data, dims=dims, amplitude=amplitude);
        
    # def tearDown(self):
        # plt.close('all')  # Close all open figures
        
    def test_constructor1(self):
        
        dimensions = [Dimension(delta=0.1), Dimension(delta=0.2)]
        data=[[0, 1, 2, 3], [4, 5, 6, 7]]
        
        sig = Signal(dims=dimensions, data=data, amp_name='dB')
        self.assertTrue(np.array_equal(sig.data, [[0, 1, 2, 3], [4, 5, 6, 7]]))
        self.assertEqual(len(sig.dims.dim_array), 2)
        self.assertEqual(sig.amplitude.name, 'dB')    
        
    def test_constructor2(self):
        data = [[[0, 1, 2, 3], [4, 5, 6, 7]], [[8, 9, 10, 11], [7, 6, 5, 4]]]
        dims = Dimension(delta=0.1, name='Time')
        sig = Signal(dims=dims, data=data, amp_name='dB')
        self.assertTrue(np.array_equal(sig.data, [[[0, 1, 2, 3], [4, 5, 6, 7]], [[8, 9, 10, 11], [7, 6, 5, 4]]]))
        self.assertEqual(len(sig.dims.dim_array), 3)
        self.assertEqual(sig.dims.dim_array[0].quantity.name, 'Time')
        self.assertEqual(sig.amplitude.name, 'dB')
        
    def test_constructor3(self):
        data = [[[0, 1, 2, 3], [4, 5, 6, 7]], [[8, 9, 10, 11], [7, 6, 5, 4]]]
        dims=[Dimension(delta=0.1), Dimension(delta=0.2)]
        sig = Signal(dims=dims, data=data, amp_name='dB')
        self.assertTrue(np.array_equal(sig.data, [[[0, 1, 2, 3], [4, 5, 6, 7]], [[8, 9, 10, 11], [7, 6, 5, 4]]]))
        self.assertEqual(len(sig.dims.dim_array), 3)
        self.assertEqual(sig.amplitude.name, 'dB')
        
    def test_plot(self):
        # Test with a new figure
        ax = self.signal.plot()

        self.assertEqual(ax.get_title(), 'Signal Data')
        self.assertEqual(ax.get_xlabel(), self.signal.dims[0].label)
        self.assertEqual(ax.get_ylabel(), self.signal.amplitude.label)

        # Test with an existing Axes object
        fig, ax2 = plt.subplots()
        ax2 = self.signal.plot(ax=ax2)

        self.assertEqual(ax2.get_title(), 'Signal Data')
        self.assertEqual(ax2.get_xlabel(), self.signal.dims[0].label)
        self.assertEqual(ax2.get_ylabel(), self.signal.amplitude.label)
    
    def test_signal_copy(self):
        # Create a Quantity object
        q = Quantity(name='length', si_unit='m')

        # Create a list of Dimension objects
        dim_array = [
            Dimension(delta=0.1, offset=1.0, quantity=q),
            Dimension(delta=0.2, offset=2.0, quantity=q),
            Dimension(delta=0.3, offset=3.0, quantity=q)
        ]

        # Create a DimensionArray object
        dims = DimensionArray(dim_array)

        # Create a numpy array
        data = np.array([
            [[1, 2], [3, 4]],
            [[5, 6], [7, 8]],
            [[9, 10], [11, 12]]
        ])

        # Create a Signal object
        s1 = Signal(data=data, dims=dims)

        # Create a copy of the Signal object
        s2 = s1.copy()

        # Check that the copied Signal object has the same state as the original
        self.assertTrue(np.array_equal(s2._data, s1._data))
        self.assertIsNot(s2._data, s1._data)
        self.assertEqual(len(s2._dims), len(s1._dims))
        for d1, d2 in zip(s1._dims.dim_array, s2._dims.dim_array):
            self.assertEqual(d2._delta, d1._delta)
            self.assertEqual(d2._offset, d1._offset)
            self.assertEqual(d2.quantity._name, d1.quantity._name)
            self.assertEqual(d2.quantity._si_unit, d1.quantity._si_unit)
            self.assertIsNot(d2, d1)
            self.assertIsNot(d2.quantity, d1.quantity)
            
    def test_crop(self):
        # Test cropping along the first dimension
        data = np.array([[1,  2,  3,  4 ],
                         [4,  5,  6,  7], 
                         [7,  8,  9,  10], 
                         [10, 11, 12, 13],
                         [1,  2,  3,  4 ],
                         [4,  5,  6,  7],
                         [7,  8,  9,  10],])
        
        dims = DimensionArray([Dimension(quantity=Quantity(name="time", si_unit="s"), delta = 0.5, offset = 2),
                               Dimension(quantity=Quantity(name="channel", si_unit="V"), delta = 2.0, offset = 3.1)])
        signal = Signal(data=data, dims=dims)

        cropped_signal = signal.crop(3, 4.5)

        expected_data = np.array([  [7,  8,  9,  10], 
                                    [10, 11, 12, 13],
                                    [1,  2,  3,  4 ]])
        expected_dims = DimensionArray([Dimension(quantity=Quantity(name="time", si_unit="s"), delta=0.5, offset=3),
                                        Dimension(quantity=Quantity(name="channel", si_unit="V"), delta=2.0, offset = 3.1)])
        
        expected_offset = 3

        np.testing.assert_array_equal(cropped_signal.data, expected_data)
        self.assertEqual(cropped_signal.ndim, 2)
        self.assertEqual(cropped_signal.shape, (3, 4))
        self.assertEqual(cropped_signal.dims[0], expected_dims[0])
        self.assertEqual(cropped_signal.dims[1], expected_dims[1])

        # Test cropping outside the range of the first dimension
        with self.assertRaises(ValueError):
            cropped_signal = signal.crop(0.5, 4.5)
            
        # Test cropping outside the range of the second dimension
        with self.assertRaises(ValueError):
            cropped_signal = signal.crop(3, 5.5)    

    def test_permute(self):
        # Test swapping the first and second dimensions
        data = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        dims = DimensionArray([Dimension(quantity=Quantity(name="time", si_unit="s"), delta=1.0),
                               Dimension(quantity=Quantity(name="channel", si_unit="V"), delta=1.0),
                               Dimension(quantity=Quantity(name="depth", si_unit="m"), delta=1.0)])
        signal = Signal(data=data, dims=dims)

        permuted_signal = signal.permute((1, 0, 2))

        expected_data = np.array([[[1, 2], [5, 6]], [[3, 4], [7, 8]]])
        expected_dims = DimensionArray([Dimension(quantity=Quantity(name="channel", si_unit="V"), delta=1.0),
                                         Dimension(quantity=Quantity(name="time", si_unit="s"), delta=1.0),
                                         Dimension(quantity=Quantity(name="depth", si_unit="m"), delta=1.0)])
        
        np.testing.assert_array_equal(permuted_signal.data, expected_data)
        self.assertEqual(permuted_signal.ndim, 3)
        self.assertEqual(permuted_signal.shape, (2, 2, 2))
        self.assertEqual(permuted_signal.dims[0], expected_dims[0])
        self.assertEqual(permuted_signal.dims[1], expected_dims[1])
        self.assertEqual(permuted_signal.dims[2], expected_dims[2])
        
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
        
    def test_quantity_copy(self):
        # Create a Quantity object
        q1 = Quantity(name='length', si_unit='m')

        # Create a copy of the Quantity object
        q2 = q1.copy()

        # Check that the copied Quantity object has the same state as the original
        self.assertEqual(q2._name, q1._name)
        self.assertEqual(q2._si_unit, q1._si_unit)
        self.assertIsNot(q2, q1)    

class TestDimension(unittest.TestCase):
    def setUp(self):
        self.dim1 = Dimension(delta=2,   offset=1, quantity=Quantity(name='Time',       si_unit='s'))
        self.dim2 = Dimension(delta=0.1, offset=0, quantity=Quantity(name='Frequency',  si_unit='Hz'))
        self.dim3 = Dimension(delta=1,   offset=0, quantity=Quantity(name='Length',     si_unit='m'))
        self.dimConvert = Dimension(delta=2.0, offset=10.0)
      
    def test_dimension_copy(self):
        # Create a Quantity object
        q = Quantity(name='length', si_unit='m')

        # Create a Dimension object
        d1 = Dimension(delta=0.1, offset=1.0, quantity=q)

        # Create a copy of the Dimension object
        d2 = d1.copy()

        # Check that the copied Dimension object has the same state as the original
        self.assertEqual(d2._delta, d1._delta)
        self.assertEqual(d2._offset, d1._offset)
        self.assertEqual(d2.quantity._name, d1.quantity._name)
        self.assertEqual(d2.quantity._si_unit, d1.quantity._si_unit)
        self.assertIsNot(d2, d1)
        self.assertIsNot(d2.quantity, d1.quantity)
    
    def test_constructor(self):
        """
        Test the two version of the constructor
        """
        
        # Create a Quantity object
        q = Quantity(name='length', si_unit='m')
        # Create a Dimension object
        d1 = Dimension(delta=0.1, offset=1.0, quantity=q)
        self.assertEqual(d1._delta,  0.1)
        self.assertEqual(d1._offset, 1)
        self.assertEqual(d1.quantity._name, 'length')
        self.assertEqual(d1.quantity._si_unit, 'm')
        
        d2 = Dimension(delta=0.2, offset=0.5, name='time', si_unit='s')
        self.assertEqual(d2._delta,  0.2)
        self.assertEqual(d2._offset, 0.5)
        self.assertEqual(d2.quantity._name, 'time')
        self.assertEqual(d2.quantity._si_unit, 's')
        
            
    def test_to_sample(self):
        value = 15.0
        sample = self.dimConvert.to_sample(value)
        self.assertEqual(sample, 2)
    
    def test_to_unitvalue(self):
        sample = 2
        value = self.dimConvert.to_unitvalue(sample)
        expected_value = 14.0
        self.assertAlmostEqual(value, expected_value)
           
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

    def test_dim_vector(self):
        vec1 = np.array([1, 3, 5, 7])
        vec2 = np.array([0, 0.1, 0.2, 0.1*3])
        vec3 = np.array([0, 1, 2, 3])
        self.assertTrue(np.array_equal(self.dim1.dim_vector(4), vec1))
        self.assertTrue(np.array_equal(self.dim2.dim_vector(4), vec2))
        self.assertTrue(np.array_equal(self.dim3.dim_vector(4), vec3))
        
        
class TestDimensionArray(unittest.TestCase):
    
    def test_empty_input(self):
        dim_array1 = DimensionArray()
        self.assertEqual(dim_array1.dim_array.size, 0)

    def test_single_dimension_input(self):
        dim1 = Dimension()
        dim_array2 = DimensionArray(dim1)
        self.assertEqual(dim_array2.dim_array.size, 1)
        self.assertIs(dim_array2.dim_array[0], dim1)

    def test_array_of_dimensions_input(self):
        dim1 = Dimension()
        dim2 = Dimension()
        dim3 = Dimension()
        dim_array3 = DimensionArray([dim1, dim2, dim3])
        self.assertEqual(dim_array3.dim_array.size, 3)
        self.assertIs(dim_array3.dim_array[0], dim1)
        self.assertIs(dim_array3.dim_array[1], dim2)
        self.assertIs(dim_array3.dim_array[2], dim3)
        
    def setUp(self):
        self.dim_array = np.array([
            Dimension(delta=1.0, offset=0.0, quantity=Quantity(si_unit='m', name='length', label='L')),
            Dimension(delta=2.0, offset=1.0, quantity=Quantity(si_unit='s', name='time', label='T')),
            Dimension(delta=0.5, offset=2.0, quantity=Quantity(si_unit='kg', name='mass', label='M'))
        ])
        self.da = DimensionArray(self.dim_array)
    
    def test_dimension_array_copy(self):
        # Create a Quantity object
        q = Quantity(name='length', si_unit='m')

        # Create a list of Dimension objects
        dim_array1 = [
            Dimension(delta=0.1, offset=1.0, quantity=q),
            Dimension(delta=0.2, offset=2.0, quantity=q),
            Dimension(delta=0.3, offset=3.0, quantity=q)
        ]

        # Create a DimensionArray object
        da1 = DimensionArray(dim_array1)

        # Create a copy of the DimensionArray object
        da2 = da1.copy()
        
        # Check that the copied DimensionArray object has the same state as the original
        self.assertEqual(len(da2), len(da1))
        for d1, d2 in zip(da1, da2):
            self.assertEqual(d2._delta, d1._delta)
            self.assertEqual(d2._offset, d1._offset)
            self.assertEqual(d2.quantity._name, d1.quantity._name)
            self.assertEqual(d2.quantity._si_unit, d1.quantity._si_unit)
            self.assertIsNot(d2, d1)
            self.assertIsNot(d2.quantity, d1.quantity)
        
    def test_find_dim_name(self):
        self.assertTrue(np.array_equal(self.da.find_dim_name('length'), [True, False, False]))
        self.assertTrue(np.array_equal(self.da.find_dim_name('time'), [False, True, False]))
        self.assertTrue(np.array_equal(self.da.find_dim_name('mass'), [False, False, True]))
        self.assertTrue(np.array_equal(self.da.find_dim_name('energy'), [False, False, False]))
        
    def test_find_dim_nameIndex(self):
        self.assertTrue(np.array_equal(self.da.find_dim_nameIndex('length'), [0]))
        self.assertTrue(np.array_equal(self.da.find_dim_nameIndex('energy'), []))
        
    def test_set_dim_unit(self):
        self.da[0].si_unit = 'cm'
        self.assertEqual(self.da[0].quantity.si_unit, 'cm')
        
        self.da[2].si_unit = 'lb';
        self.assertEqual(self.da[2].quantity.si_unit, 'lb')
    
    def test_set_dim_name(self):
        
        self.da[0].name = 'distance'
        self.assertEqual(self.da[0].name, 'distance')
        
        self.da[2].name = 'weight';
        self.assertEqual(self.da[2].quantity.name, 'weight')
        
    
    def test_set_dim_delta(self):
        self.da[0].delta = 0.1
        self.assertEqual(self.da[0].delta, 0.1)
        
        self.da[2].delta = 0.3;
        self.assertEqual(self.da[2].delta, 0.3)

    
    def test_set_dim_offset(self):
        self.da[0].offset = 0.5
        self.assertEqual(self.da[0].offset, 0.5)
        
        self.da[2].offset = 2.3;
        self.assertEqual(self.da[2].offset, 2.3)
        
    
    def test_input_array_error(self):
        with self.assertRaises(ValueError):
            DimensionArray([1, 2, 3])
            
        with self.assertRaises(ValueError):
            DimensionArray([Dimension(), 'not a dimension', Dimension()])
            
if __name__ == '__main__':
    unittest.main()