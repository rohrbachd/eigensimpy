import unittest

# Import square function
from eigensimpy.ussim.Transducers import Emitter
from eigensimpy.dsp.Signals import Signal, Dimension, DimensionArray, Quantity, Receiver
import numpy as np

class TestEmitter(unittest.TestCase):

        
    def test_emitt_signal(self):
        
        data = np.array([1, 2, 3, 4, 5, 6, 7])
        dimTime = Dimension(quantity=Quantity(name="time", si_unit="s"), delta=0.5, offset=2);
        dims = DimensionArray([dimTime])
        signal = Signal(data=data, dims=dims)
        
        # Set up an emitter at position 3
        position = np.array( [ [ 0, 3 ], 
                               [ 1, 0 ], 
                               [ 4, 5 ] ] , dtype=np.uint64)
        emitter = Emitter(position=position, signal=signal)

        # Set up a medium with the same shape as the signal
        field = np.zeros( [10, 20] )
        expected_field = field.copy();
        
        # Emit the signal at time index 2 = sample 0 
        field = emitter.emitt_signal(2, field)
        expected_field[ 0, 3] = 1;
        expected_field[ 1, 0] = 1;
        expected_field[ 4, 5] = 1;
        
        np.testing.assert_array_equal(field, expected_field)
        
        field = emitter.emitt_signal( 3.5, field)
        expected_field[ 0, 3] = 1+4;
        expected_field[ 1, 0] = 1+4;
        expected_field[ 4, 5] = 1+4;
        np.testing.assert_array_equal(field, expected_field)
        
        # Test that an error is raised if the signal has zero signals in the first dimension
        data = np.array( np.array( [1,2,3,4,5,6,7,8] ).reshape( 8, 1) )
        dims = DimensionArray([ dimTime,
                                Dimension(quantity=Quantity(name="channel", si_unit="V"), delta=2.0, offset=3.1)]) # second dimensions
        signal = Signal(data=data, dims=dims)    
        emitter = Emitter(position=position, signal=signal);
        
        field = emitter.emitt_signal( 5.5, field)
        expected_field[ 0, 3] = 5+8;
        expected_field[ 1, 0] = 5+8;
        expected_field[ 4, 5] = 5+8;
        np.testing.assert_array_equal(field, expected_field)
        
    def test_init(self):
        # Test that an error is raised if the signal has more than one dimension
        data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        dims = DimensionArray([Dimension(quantity=Quantity(name="time", si_unit="s"), delta=0.5, offset=2),
                                   Dimension(quantity=Quantity(name="channel", si_unit="V"), delta=2.0, offset=3.1)])
        signal = Signal(data=data, dims=dims)
        # Set up an emitter at position 3
        position = np.array( [ [ 0, 3 ], 
                               [ 1, 0 ], 
                               [ 4, 5] ] , dtype=np.uint64)
        
        with self.assertRaises(ValueError):
            Emitter(position=position, signal=signal)

        # Test that an error is raised if the signal has zero signals in the first dimension
        data = np.array( np.array( [1,2,3,4,5,6,7,8] ).reshape( 1, 8) )
        dims = DimensionArray([Dimension(quantity=Quantity(name="time", si_unit="s"), delta=0.5, offset=2),
                               Dimension(quantity=Quantity(name="channel", si_unit="V"), delta=2.0, offset=3.1)])
        signal = Signal(data=data, dims=dims)

        with self.assertRaises(ValueError):
            Emitter(position=position, signal=signal)
            
            
    def test_record_signal(self):
    # Set up a receiver at position 3
    
        position = np.array([[0, 3], [1, 0], [4, 5]], dtype=np.uint64)
        dims = DimensionArray([Dimension(quantity=Quantity(name="time", si_unit="s"), delta=0.5, offset=2), ])
        data = np.zeros(8,1);
        
        receiver = Receiver(position=position, dims=dims)

        # Set up a medium with the same shape as the signal
        field = np.zeros([10, 20])
        field[0, 3 ] = 1.0
        field[1, 0 ] = 2.0
        field[4, 5 ] = 3.0
        
        # Record the signal at time index 2 = sample 0 
        field = receiver.record_signal(2, field)
        expected_signal_data = np.array([1.0])
        np.testing.assert_allclose(receiver.signal.data, expected_signal_data)

        # Record the signal at time index 3 = sample 1
        field = receiver.record_signal(3, field)
        expected_signal_data = np.array([2.0])
        np.testing.assert_allclose(receiver.signal.data, expected_signal_data)

        # Record the signal at time index 4 = sample 2
        field = receiver.record_signal(4, field)
        expected_signal_data = np.array([3.0])
        np.testing.assert_allclose(receiver.signal.data, expected_signal_data)        