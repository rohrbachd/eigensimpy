from eigensimpy.ureg import unit
from eigensimpy.ussim.Media import IsotropicAcousticMedia 
from eigensimpy.dsp.Signals import Signal

import numpy as np

class ViscousMaterial:
    
    def __init__(self, **kwargs) -> None:
        self._comp_sos    = self._get_quantity(kwargs.get('comp_sos', 0), unit.meter / unit.second)
        self._shear_sos   = self._get_quantity(kwargs.get('shear_sos', 0), unit.meter / unit.second)
        self._density     = self._get_quantity(kwargs.get('density', 0), unit.kilogram / unit.meter**3)
        self._comp_atten  = self._get_quantity(kwargs.get('comp_atten', 0), unit.neper)
        self._shear_atten = self._get_quantity(kwargs.get('shear_atten', 0), unit.neper)
        
    def _get_quantity(self, value, si_unit):
        if isinstance(value, (int, float)):
            return unit.Quantity(value, si_unit)
        elif isinstance(value, unit.Quantity):
            return value.to(si_unit)
        else:
            raise ValueError(f"Invalid input for {si_unit}. Provide a value with the correct unit, or a plain number to use the default unit.")

    def to_media(self, size, dimension=None):
    
        comp_sos_data    = np.zeros(size) + self.comp_sos.magnitude
        shear_sos_data   = np.zeros(size) + self.shear_sos.magnitude
        density_data     = np.zeros(size) + self.density.magnitude
        comp_atten_data  = np.zeros(size) + self.comp_atten.magnitude
        shear_atten_data = np.zeros(size) + self.shear_atten.magnitude
        
        amp_names = {
            "comp_sos":     "speed of sound",
            "shear_sos":    "speed of sound",
            "density":      "density",
            "comp_atten":   "attenuation",
            "shear_atten":  "attenuation"
        }

        amp_si_units = {
            "comp_sos":     str(self.comp_sos.units),
            "shear_sos":    str(self.shear_sos.units),
            "density":      str(self.density.units),
            "comp_atten":   str(self.comp_atten.units),
            "shear_atten":  str(self.shear_atten.units)
        }

        if dimension:
            comp_sos    = Signal(data=comp_sos_data,    dims=dimension, amp_name=amp_names["comp_sos"],     amp_si_unit=amp_si_units["comp_sos"])
            shear_sos   = Signal(data=shear_sos_data,   dims=dimension, amp_name=amp_names["shear_sos"],    amp_si_unit=amp_si_units["shear_sos"])
            density     = Signal(data=density_data,     dims=dimension, amp_name=amp_names["density"],      amp_si_unit=amp_si_units["density"])
            comp_atten  = Signal(data=comp_atten_data,  dims=dimension, amp_name=amp_names["comp_atten"],   amp_si_unit=amp_si_units["comp_atten"])
            shear_atten = Signal(data=shear_atten_data, dims=dimension, amp_name=amp_names["shear_atten"],  amp_si_unit=amp_si_units["shear_atten"])
        else:
            comp_sos    = Signal(data=comp_sos_data,    amp_name=amp_names["comp_sos"],     amp_si_unit=amp_si_units["comp_sos"])
            shear_sos   = Signal(data=shear_sos_data,   amp_name=amp_names["shear_sos"],    amp_si_unit=amp_si_units["shear_sos"])
            density     = Signal(data=density_data,     amp_name=amp_names["density"],      amp_si_unit=amp_si_units["density"])
            comp_atten  = Signal(data=comp_atten_data,  amp_name=amp_names["comp_atten"],   amp_si_unit=amp_si_units["comp_atten"])
            shear_atten = Signal(data=shear_atten_data, amp_name=amp_names["shear_atten"],  amp_si_unit=amp_si_units["shear_atten"])

        isotropic_acoustic_media = IsotropicAcousticMedia(
            comp_sos = comp_sos,
            shear_sos = shear_sos,
            density = density,
            comp_atten = comp_atten,
            shear_atten = shear_atten)

        return isotropic_acoustic_media

    @property
    def comp_sos(self):
        return self._comp_sos

    @property
    def shear_sos(self):
        return self._shear_sos

    @property
    def density(self):
        return self._density

    @property
    def comp_atten(self):
        return self._comp_atten

    @property
    def shear_atten(self):
        return self._shear_atten

