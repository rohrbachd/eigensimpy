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
    
        comp_sos_data    = np.zeros(size) + self.comp_sos
        shear_sos_data   = np.zeros(size) + self.shear_sos
        density_data     = np.zeros(size) + self.density
        comp_atten_data  = np.zeros(size) + self.comp_atten
        shear_atten_data = np.zeros(size) + self.shear_atten

        if dimension:
            comp_sos    = Signal(data=comp_sos_data, dims=dimension)
            shear_sos   = Signal(data=shear_sos_data, dims=dimension)
            density     = Signal(data=density_data, dims=dimension)
            comp_atten  = Signal(data=comp_atten_data, dims=dimension)
            shear_atten = Signal(data=shear_atten_data, dims=dimension)
        else:
            comp_sos    = Signal(data=comp_sos_data)
            shear_sos   = Signal(data=shear_sos_data)
            density     = Signal(data=density_data)
            comp_atten  = Signal(data=comp_atten_data)
            shear_atten = Signal(data=shear_atten_data)

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

