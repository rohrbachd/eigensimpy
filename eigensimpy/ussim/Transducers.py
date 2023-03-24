

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

        for i in range(self.num_pos):
            val = self.signal.value_at(time)
            pos = tuple(self.position[i, :].astype(int))

            field[pos] = field[pos] + val


        return field
    
class Receiver:
    
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

    def record_signal(self, time: float, field: np.ndarray) -> np.ndarray:

        value = 0;
        for i in range(self.num_pos):
            
            pos = tuple(self.position[i, :].astype(int))
            value = value + field( pos )
            
        self.signal.set_value_at( value, time)

        return field    