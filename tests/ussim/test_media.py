import unittest

# Import square function
from eigensimpy.ussim.Materials import ViscousMaterial
from eigensimpy.ussim.Media import IsotropicAcousticMedia, IsotropicElasticMedia
from eigensimpy.dsp.Signals import Signal, Dimension
from eigensimpy.ureg import unit, neper
import numpy as np


class TestIsotropicAcousticMedia(unittest.TestCase):

    def setUp(self):
        
        material = ViscousMaterial(comp_sos = 1500, 
                                   shear_sos = 1000, 
                                   density = 1000, 
                                   comp_atten = 0.5, 
                                   shear_atten = 0.3)
        self.media = material.to_media([10, 25])

    def test_compute_chi(self):
        chi = self.media.compute_chi()
        self.assertIsNotNone(chi)
        self.assertIsInstance(chi, Signal)
        self.assertEqual(chi.shape, (10,25))

    def test_compute_eta(self):
        eta = self.media.compute_eta()
        self.assertIsNotNone(eta)
        self.assertIsInstance(eta, Signal)
        self.assertEqual(eta.shape, (10,25))
        
    def test_convert_to_elastic(self):
        elastic = self.media.convert_to_elastic()
        self.assertIsNotNone(elastic)
        self.assertIsInstance(elastic, IsotropicElasticMedia)
        
    def test_db2neper(self):
        atten_db_mhz_cm = 1
        power = 1
        atten_np = IsotropicAcousticMedia.db2neper(atten_db_mhz_cm, power)
        self.assertIsNotNone(atten_np)
        self.assertAlmostEqual(atten_np, 1.83233e-06, delta=1e-5)
         
        
    # def test_invalid_input(self):
    #     with self.assertRaises(TypeError):
    #         IsotropicAcousticMedia(comp_sos="invalid", shear_sos=1000, density=1000, comp_atten=0.5, shear_atten=0.3)
