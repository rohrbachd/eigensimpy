import unittest

# Import square function
from eigensimpy.ussim.Materials import ViscousMaterial
from eigensimpy.dsp.Signals import Signal, Dimension
from eigensimpy.ureg import unit, neper
import numpy as np

#elp(Quantity)


class TestViscous(unittest.TestCase):
    
    def test_to_media(self):
        size = (3, 3)
        dimension = Dimension(delta=0.1, name='Time', si_unit='s')
        viscous = ViscousMaterial(comp_sos=1500, shear_sos=0, density=1000, comp_atten=0, shear_atten=0, power=1)

        medium = viscous.to_media(size, dimension)

        for attr in ['comp_sos', 'shear_sos', 'density', 'comp_atten', 'shear_atten']:
            
            signal = getattr(medium, attr)
            self.assertIsInstance(signal, Signal)
            self.assertEqual(signal.data.shape, size)
            self.assertEqual(signal.dims.dim_array[0], dimension)
            material = getattr(viscous, attr)
            self.assertTrue(np.all(signal.data == material.magnitude))
            
    def test_constructor_default_values(self):
        viscous = ViscousMaterial()
        self.assertEqual(viscous.comp_sos, 0 * unit.meter / unit.second)
        self.assertEqual(viscous.shear_sos, 0 * unit.meter / unit.second)
        self.assertEqual(viscous.density, 0 * unit.kilogram / unit.meter**3)
        self.assertEqual(viscous.comp_atten, neper(0))
        self.assertEqual(viscous.shear_atten, neper(0))

    def test_constructor_with_units(self):
        viscous = ViscousMaterial(comp_sos=1500 * unit.meter / unit.second, shear_sos=3000 * unit.meter / unit.second,
                          density=1000 * unit.kilogram / unit.meter**3, comp_atten=neper(0.5),
                          shear_atten=neper(1), power=100 * unit.watt)
        self.assertEqual(viscous.comp_sos, 1500 * unit.meter / unit.second)
        self.assertEqual(viscous.shear_sos, 3000 * unit.meter / unit.second)
        self.assertEqual(viscous.density, 1000 * unit.kilogram / unit.meter**3)
        self.assertEqual(viscous.comp_atten, neper(0.5))
        self.assertEqual(viscous.shear_atten, neper(1))

    def test_constructor_without_units(self):
        viscous = ViscousMaterial(comp_sos=1500, shear_sos=3000, density=1000, comp_atten=0.5, shear_atten=1, power=100)
        self.assertEqual(viscous.comp_sos, 1500 * unit.meter / unit.second)
        self.assertEqual(viscous.shear_sos, 3000 * unit.meter / unit.second)
        self.assertEqual(viscous.density, 1000 * unit.kilogram / unit.meter**3)
        self.assertEqual(viscous.comp_atten, unit.Quantity(0.5, unit.neper))
        self.assertEqual(viscous.shear_atten, unit.Quantity(1, unit.neper))
        
if __name__ == "__main__":
    unittest.main()