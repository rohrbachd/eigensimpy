from typing import Optional, List
import numpy as np

#   field:
#
#    [ - - -  .. - *************************************** - .. - - - ]
#    [ - - -  .. - *************************************** - .. - - - ]
#                       :
#    [ - - -  .. - *************************************** - .. - - - ]
#    [ o o o  .. o xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx o .. o o o ]
#    [ o o o  .. o xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx o .. o o o ]
#                       :
#    [ o o o  .. o xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx o .. o o o ]   
#    [ o o o  .. o xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx o .. o o o ]
#    [ - - -  .. - *************************************** - .. - - - ]
#    [ - - -  .. - *************************************** - .. - - - ]
#                       :
#    [ - - -  .. - *************************************** - .. - - - ]
#
#
#  Wheere o and * are pml layers and x is data. *** = pml_d1 and ooo is pml_d2
#  -- is where pmld1 and pml2 overlap

class PMLSettings:
    def __init__(self, **kwargs):
       
        self.attenuation : float = kwargs.get("attenuation", 0.0)
        self.num_elements : float = kwargs.get("num_elements", 0.0)
        self.speed_of_sound: float = kwargs.get("speed_of_sound", 0.0)
        self.delta_x : float = kwargs.get("delta_x", 0.0)
        self.delta_t : float = kwargs.get("delta_t", 0.0)
    
    
class SimplePML:
    def __init__(self, settings: Optional[PMLSettings] = None, **kwargs):
        if settings is None:
            settings = PMLSettings(**kwargs)

        self.settings = settings
        is_staggered = False
        self.pml_origin_cache = self._compute_pml1d(is_staggered)
        is_staggered = True
        self.pml_staggered_cache = self._compute_pml1d(is_staggered)

    @staticmethod
    def example():
        settings = PMLSettings(delta_t=2.8e-9, delta_x=1.6e-5, speed_of_sound=1500, num_elements=30, attenuation=2)
        return SimplePML(settings)

    def create_multi_axial_pml(self, ratio):
        maxial_settings = self.settings
        maxial_settings.attenuation = maxial_settings.attenuation * ratio
        return SimplePML(maxial_settings)

    def apply_pml(self, field: np.ndarray, is_stag: Optional[List[bool]] = None):
        
        if is_stag is None:
            is_stag = []

        nd = len(field.shape)
        is_stag = self._validate_stag(is_stag, nd)

        pml_origin = self.pml_origin_cache
        pml_stg = self.pml_staggered_cache

        N = len(pml_origin)

        self._must_be_2D(nd)

        if is_stag[0]:
            field = apply_pml_d1_helper(field, pml_stg, N)
        else:
            field = apply_pml_d1_helper(field, pml_origin, N)

        if is_stag[1]:
            field = apply_pml_d2_helper(field, pml_stg, N)
        else:
            field = apply_pml_d2_helper(field, pml_origin, N)

        return field
    
    def apply_pml_d1(self, field, is_stag:bool):
        
        nd = len(field.shape)

        pml_origin = self.pml_origin_cache
        pml_stg = self.pml_staggered_cache

        N = len(pml_origin)

        self._must_be_2D(nd)

        if is_stag[0]:
            field = apply_pml_d1_helper(field, pml_stg, N)
        else:
            field = apply_pml_d1_helper(field, pml_origin, N)

        return field
    
    
    def apply_pml_d2(self, field, is_stag:bool):
        
        nd = len(field.shape)
        is_stag = self._validate_stag(is_stag, nd)

        pml_origin = self.pml_origin_cache
        pml_stg = self.pml_staggered_cache

        N = len(pml_origin)

        self._must_be_2D(nd)

        if is_stag[1]:
            field = apply_pml_d2_helper(field, pml_stg, N)
        else:
            field = apply_pml_d2_helper(field, pml_origin, N)

        return field

    def _must_be_2D(self, nd):
        if nd != 2:
            raise ValueError('SimplePML:non2DNotSupported: non2D pml not supported yet')

    def _validate_stag(self, is_stag, nd):
        if not is_stag:
            is_stag = [False] * nd

        self._must_be_valid_stag(is_stag, nd)
        return is_stag

    def _must_be_valid_stag(self, is_stag, nd):
        if len(is_stag) < nd:
            raise ValueError('SimplePML:invalidNumel: isStag must have at least as many elements as dimensions in field')
    

    def _compute_pml1d(self, is_staggered: bool):
        settings = self.settings
        a = settings.attenuation
        N = settings.num_elements
        c = settings.speed_of_sound

        dx = settings.delta_x
        dt = settings.delta_t

        if is_staggered:
            x = np.arange(1, N + 1)
        else:
            x = np.arange(1.5, N + 1.5)

        coeff = a * c / dx
        pml = coeff * (x / N) ** 4
        pml = np.exp(-pml * (dt / 2))

        return pml
    
def apply_pml_d1_helper(field, pml, N):
    """Apply PML in the first dimension of the field.
    
    Args:
        field (numpy.ndarray): The field to which the PML will be applied.
        pml (numpy.ndarray): The 1-dimensional PML array.
        N (int): The number of elements in the PML array.

    Returns:
        numpy.ndarray: The field after applying the PML in the first dimension.
    """
    
    # from -N to END of field in 1st dimension  
    field[-N:, :] *= pml[:, None]
    # from 0 to N of field in 1st dimension  
    field[:N, :] *= pml[::-1, None]
    return field

def apply_pml_d2_helper(field, pml, N):
    """Apply PML in the second dimension of the field.
    
    Args:
        field (numpy.ndarray): The field to which the PML will be applied.
        pml (numpy.ndarray): The 1-dimensional PML array.
        N (int): The number of elements in the PML array.

    Returns:
        numpy.ndarray: The field after applying the PML in the second dimension.
    """
    
    # from -N to END of field in the 2nd dimension  
    field[:, -N:] *= pml[None, :]
    # from 0 to N of field in the 2nd dimension      
    field[:, :N] *= pml[None, ::-1]
    return field