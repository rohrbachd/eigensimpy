

import numpy as np
from eigensimpy.dsp.Signals import Signal

class Emitter:
    
    def __init__(self, **kwargs):
        position : np.ndarray = kwargs.pop('position', None)
        signal : Signal= kwargs.pop('signal', None)

        if position is None:
            raise ValueError("Please provide a 'position' parameter.")
        if signal is None:
            raise ValueError("Please provide a 'signal' parameter.")
        
        num_signals = signal.num_signals(0)
        if num_signals != 1:
            raise ValueError("Signal must have only one signal in the first dimension")
        
        self.position = position
        self.signal = signal

        self.update_num_pos()

    @property
    def position(self) -> np.ndarray:
        return self._position

    @position.setter
    def position(self, value: np.ndarray) -> None:
        self._position = value
        self.update_num_pos()

    def update_num_pos(self) -> None:
        self.num_pos = self.position.shape[0]

    def emitt_signal(self, time: float, field: np.ndarray) -> np.ndarray:
        source = np.zeros_like(field)

        for i in range(self.num_pos):
            val = self.signal.value_at(time, 0)
            pos = tuple(self.position[i, :].astype(int))

            source[pos] = val

        field = field + source

        return field
    