import unittest
from eigensimpy.ureg import unit
from eigensimpy.dsp.Signals import Signal,Dimension, Quantity, DimensionArray
from eigensimpy.ussim.Fields import AcousticField2D

import numpy as np

class TestAcousticField2D(unittest.TestCase):

    def setUp(self):
        
        self.dims = DimensionArray([Dimension(quantity=Quantity(name="time", si_unit="s"), delta = 0.5, offset = 2),
                               Dimension(quantity=Quantity(name="channel", si_unit="V"), delta = 2.0, offset = 3.1)])
        
        self.af = AcousticField2D(field_size=(3, 5), dimensions=self.dims)

    def test_init(self):
        self.assertIsInstance(self.af.vel1.data, np.ndarray)
        self.assertEqual(self.af.vel1.data.shape, (3, 5))
        self.assertEqual(self.af.vel1.data.dtype, np.float64)
        self.assertEqual(self.af.vel1.dims,  self.dims)
        
        self.assertIsInstance(self.af.vel2.data, np.ndarray)
        self.assertEqual(self.af.vel2.data.shape, (3, 5))
        self.assertEqual(self.af.vel2.data.dtype, np.float64)
        self.assertEqual(self.af.vel2.dims,  self.dims)
        
        self.assertIsInstance(self.af.stress11.data, np.ndarray)
        self.assertEqual(self.af.stress11.data.shape, (3, 5))
        self.assertEqual(self.af.stress11.data.dtype, np.float64)
        self.assertEqual(self.af.stress11.dims,  self.dims)
        
        self.assertIsInstance(self.af.stress22.data, np.ndarray)
        self.assertEqual(self.af.stress22.data.shape, (3, 5))
        self.assertEqual(self.af.stress22.data.dtype, np.float64)
        self.assertEqual(self.af.stress22.dims, self.dims)
        
        self.assertIsInstance(self.af.stress12.data, np.ndarray)
        self.assertEqual(self.af.stress12.data.shape, (3, 5))
        self.assertEqual(self.af.stress12.data.dtype, np.float64)
        self.assertEqual(self.af.stress12.dims, self.dims)

    # def test_init_with_defaults(self):
        
    #     af = AcousticField2D()
    #     self.assertEqual(af.vel1.name, 'data')
    #     self.assertIsInstance(af.vel1.data, np.ndarray)
    #     self.assertEqual(af.vel1.data.shape, (0,))
    #     self.assertEqual(af.vel1.data.dtype, np.float64)
    #     self.assertEqual(af.vel1.dims, 0)
        
    #     self.assertEqual(af.vel2.name, 'data')
    #     self.assertIsInstance(af.vel2.data, np.ndarray)
    #     self.assertEqual(af.vel2.data.shape, (0,))
    #     self.assertEqual(af.vel2.data.dtype, np.float64)
    #     self.assertEqual(af.vel2.dims, 0)
        
    #     self.assertEqual(af.stress11.name, 'data')
    #     self.assertIsInstance(af.stress11.data, np.ndarray)
    #     self.assertEqual(af.stress11.data.shape, (0,))
    #     self.assertEqual(af.stress11.data.dtype, np.float64)
    #     self.assertEqual(af.stress11.dims, 0)
        
    #     self.assertEqual(af.stress22.name, 'data')
    #     self.assertIsInstance(af.stress22.data, np.ndarray)
    #     self.assertEqual(af.stress22.data.shape, (0,))
    #     self.assertEqual(af.stress22.data.dtype, np.float64)
    #     self.assertEqual(af.stress22.dims, 0)