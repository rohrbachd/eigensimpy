# import unittest

# # Import square function
# from eigensimpy.ussim.Media import IsotropicAcousticMedia, IsotropicElasticMedia
# from eigensimpy.dsp.Signals import Signal, Dimension
# from eigensimpy.ureg import unit, neper
# import numpy as np


# class TestIsotropicAcousticMedia(unittest.TestCase):

#     def setUp(self):
#         self.media = IsotropicAcousticMedia(comp_sos=1500, shear_sos=1000, density=1000, comp_atten=0.5, shear_atten=0.3)

#     def test_compute_chi(self):
#         chi = self.media.compute_chi()
#         self.assertIsNotNone(chi)
#         self.assertIsInstance(chi, Signal)

#     def test_compute_eta(self):
#         eta = self.media.compute_eta()
#         self.assertIsNotNone(eta)
#         self.assertIsInstance(eta, Signal)

#     def test_convert_to_lame(self):
#         lame = self.media.convert_to_lame()
#         self.assertIsNotNone(lame)
#         self.assertIsInstance(lame, IsotropicElasticMedia)

#     def test_db2neper(self):
#         atten_db_mhz_cm = 1
#         power = 1
#         atten_np = IsotropicAcousticMedia.db2neper(atten_db_mhz_cm, power)
#         self.assertIsNotNone(atten_np)
#         self.assertIsInstance(atten_np, Signal)

#     def test_invalid_input(self):
#         with self.assertRaises(TypeError):
#             IsotropicAcousticMedia(comp_sos="invalid", shear_sos=1000, density=1000, comp_atten=0.5, shear_atten=0.3)
