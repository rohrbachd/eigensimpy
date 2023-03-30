import unittest

# Import square function
from eigensimpy.ussim.Boundaries import SimplePML, PMLSettings
from eigensimpy.dsp.Signals import Signal, Dimension
import numpy as np

class TestPML(unittest.TestCase):
    
    def test_apply_pml(self):
        pml_settings = PMLSettings(delta_t=2.8e-9, delta_x=1.6e-5, speed_of_sound=1500, num_elements=30, attenuation=2)
        pml = SimplePML(pml_settings)

        field = np.ones((100, 100))
        original_field = field.copy()

        modified_field = pml.apply_pml(field)

        N = pml_settings.num_elements
        pml_origin = pml.pml_origin_cache

        # Test the top and bottom sections of the field in the first dimension
        self.assertTrue(np.allclose(modified_field[:N,  31:70], pml_origin[::-1, None]) )
        self.assertTrue(np.allclose(modified_field[-N:, 31:70], pml_origin[:, None]))

        # Test the left and right sections of the field in the second dimension
        self.assertTrue(np.allclose(modified_field[31:70, :N], pml_origin[None, ::-1]))
        self.assertTrue(np.allclose(modified_field[31:70, -N:], pml_origin[None, :]))

        # Test that the center parts of the modified_field should match the original field
        self.assertTrue(np.allclose(modified_field[N:-N, N:-N], original_field[N:-N, N:-N]))

        
        

if __name__ == "__main__":
    unittest.main()