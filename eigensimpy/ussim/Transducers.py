

import numpy as np
from eigensimpy.dsp.Signals import Signal

class EmitterSet2D:
    
    def __init__(self, **kwargs):
        self.emitter_vel1 : Emitter = kwargs.pop('emitter_vel1', [])
        self.emitter_vel2 : Emitter = kwargs.pop('emitter_vel2', [])
        self.emitter_stress11 : Emitter = kwargs.pop('emitter_stress11', [])
        self.emitter_stress22 : Emitter = kwargs.pop('emitter_stress22', [])
        self.emitter_stress12 : Emitter = kwargs.pop('emitter_stress12', [])
        
        self._validate_emitters()

    def emit_stress(self, ti, fieldBuffer):
        fieldBuffer.stress11 = self.emit_stress11(ti, fieldBuffer.stress11)
        fieldBuffer.stress22 = self.emit_stress22(ti, fieldBuffer.stress22)
        fieldBuffer.stress12 = self.emit_stress12(ti, fieldBuffer.stress12)
        return fieldBuffer

    def emit_velocity(self, ti, fieldBuffer):
        fieldBuffer.vel1 = self.emit_vel1(ti, fieldBuffer.vel1)
        fieldBuffer.vel2 = self.emit_vel2(ti, fieldBuffer.vel2)
        return fieldBuffer
    
    def draw_on_map(self, map, type):
        emitter = getattr(self, type)
        if emitter:
            for e in emitter:
                pos = e.position
                for pi in range(pos.shape[0]):
                    map[pos[pi, 0], pos[pi, 1]] = 1

    def has_no_emitters(self):
        return len(self.emitters) == 0
        
    @property
    def emitters(self):
        return [self.emitter_vel1, self.emitter_vel2, self.emitter_stress11,
                self.emitter_stress22, self.emitter_stress12]
        
    @property
    def time_unit(self):
        emitter = self.emitters
        if emitter:
            return emitter[0].signal.dims[0].quantity.si_unit
        else:
            return ""
        
    @property
    def delta(self):
        emitter = self.emitters
        if emitter:
            return emitter[0].signal.dims[0].delta
        else:
            return 1
        
    @property
    def offset(self):
        emitter = self.emitters
        if emitter:
            return emitter[0].signal.dims[0].offset
        else:
            return 0
        
    def _validate_emitters(self):
        emitter = self.emitters
        nEmitters = len(emitter)
        
        for i in range(1, nEmitters):
            sig = emitter[i].signal
            if sig.dims[0].delta != self.delta or sig.dims[0].offset != self.offset or sig.dims[0].quantity.si_unit != self.time_unit:
                raise ValueError("All signals must match in delta, offset, and si_unit")
         
    def emit_vel1(self, time: float, field: np.ndarray) -> np.ndarray:
        field = self.emitter_vel1.emitt_signal(time, field)
        return field

    def emit_vel2(self, time: float, field: np.ndarray) -> np.ndarray:
        field = self.emitter_vel2.emitt_signal(time, field)
        return field

    def emit_stress11(self, time: float, field: np.ndarray) -> np.ndarray:
        field = self.emitter_stress11.emitt_signal(time, field)
        return field

    def emit_stress22(self, time: float, field: np.ndarray) -> np.ndarray:
        field = self.emitter_stress22.emitt_signal(time, field)
        return field

    def emit_stress12(self, time: float, field: np.ndarray) -> np.ndarray:
        field = self.emitter_stress12.emitt_signal(time, field)
        return field        
           
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

    def record_signal(self, time: float, field: np.ndarray) -> None:

        value = 0;
        for i in range(self.num_pos):
            
            pos = tuple(self.position[i, :].astype(int))
            value = value + field[pos]
            
        self.signal.set_value_at( value, time)   