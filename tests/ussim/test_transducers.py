import unittest

# Import square function
from eigensimpy.ussim.Transducers import Emitter
from eigensimpy.dsp.Signals import Signal, Dimension, DimensionArray, Quantity
import numpy as np

class TestEmitter(unittest.TestCase):

    def setUp(self):
        # Set up a 1D signal
        self.data = np.array([1, 2, 3, 4, 5, 6, 7])
        self.dims = DimensionArray([Dimension(quantity=Quantity(name="time", si_unit="s"), delta=0.5, offset=2)])
        self.signal = Signal(data=self.data, dims=self.dims)
        
        # Set up an emitter at position 3
        self.position = np.array([[3]], dtype=np.uint64)
        self.emitter = Emitter(position=self.position, signal=self.signal)

    def test_emitt_signal(self):
        # Set up a medium with the same shape as the signal
        field = np.zeros_like(self.data)
        
        
        # Emit the signal at time index 2
        self.emitter.emitt_signal(2, medium.field)
        
        # Check that the medium field has been updated
        expected_field = np.array([0, 0, 3, 0, 0, 0, 0])
        np.testing.assert_array_equal(medium.field, expected_field)

    def test_init(self):
        # Test that an error is raised if the signal has more than one dimension
        data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        dims = DimensionArray([Dimension(quantity=Quantity(name="time", si_unit="s"), delta=0.5, offset=2),
                                   Dimension(quantity=Quantity(name="channel", si_unit="V"), delta=2.0, offset=3.1)])
        signal = Signal(data=data, dims=dims)

        with self.assertRaises(ValueError):
            Emitter(position=self.position, signal=signal)

        # Test that an error is raised if the signal has zero signals in the first dimension
        data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        dims = DimensionArray([Dimension(quantity=Quantity(name="time", si_unit="s"), delta=0.5, offset=2),
                                   Dimension(quantity=Quantity(name="channel", si_unit="V"), delta=2.0, offset=3.1)])
        signal = Signal(data=data, dims=dims)

        with self.assertRaises(ValueError):
            Emitter(position=self.position, signal=signal)

        # Test that an error is raised if the signal has more than one signal in the first dimension
        data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        dims = DimensionArray([Dimension(quantity=Quantity(name="time", si_unit="s"), delta=0.5, offset=2),
                               Dimension(quantity=Quantity(name="channel", si_unit="V"), delta=2.0, offset=3.1)])
        signal = Signal(data=data, dims=dims)

        with self.assertRaises(ValueError):
            Emitter(position=self.position, signal=signal)