import unittest

# Import square function
from eigensimpy.ussim.Materials import ViscousMaterial
from eigensimpy.ussim.Media import IsotropicAcousticMedia, IsotropicElasticMedia
from eigensimpy.dsp.Signals import Signal
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

    def test_creation(self):
        
        media = IsotropicAcousticMedia()
        self.assertEqual(media.comp_atten.shape,    (0,0))
        self.assertEqual(media.shear_atten.shape,   (0,0))
        self.assertEqual(media.density.shape,       (0,0))
        self.assertEqual(media.shear_sos.shape,     (0,0))
        self.assertEqual(media.comp_sos.shape,      (0,0))
        
        media = IsotropicAcousticMedia(default_shape=(10,15))
        
        self.assertEqual(media.comp_atten.shape,    (10,15))
        self.assertEqual(media.shear_atten.shape,   (10,15))
        self.assertEqual(media.density.shape,       (10,15))
        self.assertEqual(media.shear_sos.shape,     (10,15))
        self.assertEqual(media.comp_sos.shape,      (10,15))
        
        # ToDo continue
        media = IsotropicAcousticMedia( density=Signal( data=np.zeros( [3,11] )) )
        
        self.assertEqual(media.comp_atten.shape,    (3,11))
        self.assertEqual(media.shear_atten.shape,   (3,11))
        self.assertEqual(media.density.shape,       (3,11))
        self.assertEqual(media.shear_sos.shape,     (3,11))
        self.assertEqual(media.comp_sos.shape,      (3,11))
        
        
        # ToDo continue
        media = IsotropicAcousticMedia( density=Signal( data=np.zeros( [4,11] )),
                                        comp_atten=Signal( data=np.zeros( [4,11] ) ) )
        
        self.assertEqual(media.comp_atten.shape,    (4,11))
        self.assertEqual(media.shear_atten.shape,   (4,11))
        self.assertEqual(media.density.shape,       (4,11))
        self.assertEqual(media.shear_sos.shape,     (4,11))
        self.assertEqual(media.comp_sos.shape,      (4,11))
        
        with self.assertRaises(ValueError):
            IsotropicAcousticMedia( density=Signal( data=np.zeros( [4,11] )),
                                    comp_sos=Signal( data=np.zeros( [4,11] )),
                                    comp_atten=Signal( data=np.zeros( [3,10] ) ) )

        
        
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
        
        self.assertEqual(elastic.chi_atten.shape, (10,25))
        self.assertEqual(elastic.eta_atten.shape, (10,25))
        self.assertEqual(elastic.density.shape, (10,25))
        self.assertEqual(elastic.shear_modulus.shape, (10,25))
        self.assertEqual(elastic.comp_modulus.shape, (10,25))
        
        np.testing.assert_allclose(elastic.chi_atten.data, 0.63428, rtol=1e-5, atol=0)
        np.testing.assert_allclose(elastic.eta_atten.data, 0.17498, rtol=1e-4, atol=0)
        np.testing.assert_allclose(elastic.density.data, 1000, rtol=0, atol=0)
        np.testing.assert_allclose(elastic.shear_modulus.data, 1e9, rtol=0, atol=0)
        np.testing.assert_allclose(elastic.comp_modulus.data, 250000000, rtol=0, atol=0)
        
        
    def test_db2neper(self):
        atten_db_mhz_cm = 1
        power = 1
        atten_np = IsotropicAcousticMedia.db2neper(atten_db_mhz_cm, power)
        self.assertIsNotNone(atten_np)
        self.assertAlmostEqual(atten_np, 1.83233e-06, delta=1e-5)
         
        
    # def test_invalid_input(self):
    #     with self.assertRaises(TypeError):
    #         IsotropicAcousticMedia(comp_sos="invalid", shear_sos=1000, density=1000, comp_atten=0.5, shear_atten=0.3)


if __name__ == "__main__":
    unittest.main()