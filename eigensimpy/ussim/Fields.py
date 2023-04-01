
from eigensimpy.dsp.Signals import Signal, DimensionArray

import numpy as np

class AcousticField2D:
    
    def __init__(self, **kwargs) -> None:
        
        size = kwargs.get('field_size', 0)
        dimensions : DimensionArray = kwargs.get('dimensions', 0)
        
        self._initField(size, dimensions)
        

    def _initField(self, size, dimensions):
        
        data = np.zeros(size)
        
        self.vel1 = Signal(data = data.copy(),dims =  dimensions )
        self.vel2 = Signal(data = data.copy(),dims =  dimensions )
        self.stress11 = Signal(data = data.copy(),dims =  dimensions )
        self.stress22 = Signal(data = data.copy(),dims =  dimensions )
        self.stress12 = Signal(data = data,       dims =  dimensions )