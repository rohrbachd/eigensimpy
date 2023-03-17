import unittest
import numpy as np

# Import square function
from eigensimpy.dsp.Signals import Signal, Quantity, Dimension, DimensionArray
from eigensimpy.dsp.SignalFactories import PulseFactory

import matplotlib.pyplot as plt
#elp(Quantity)

class SignalFactoryTests(unittest.TestCase):
    
    def test_simple_signal(self):
        pf = PulseFactory();
        signal = pf.calc_pulse('simple1') 
        
        ax = signal.plot()
        self.assertEqual(ax.get_title(), 'Signal Data')