

import numpy as np
from eigensimpy.dsp.Signals import Signal

class ReceiverSet2D:
    
    def __init__(self, **kwargs):
        self.receiver_vel1 : Receiver = kwargs.pop('receiver_vel1', [])
        self.receiver_vel2 : Receiver = kwargs.pop('receiver_vel2', [])
        self.receiver_stress11 : Receiver = kwargs.pop('receiver_stress11', [])
        self.receiver_stress22 : Receiver = kwargs.pop('receiver_stress22', [])
        self.receiver_stress12 : Receiver = kwargs.pop('receiver_stress12', [])
        
        self._transducer_manager = TransducerManager( [receiver for receiver in [   self.receiver_vel1, 
                                                                                    self.receiver_vel2, 
                                                                                    self.receiver_stress11, 
                                                                                    self.receiver_stress22, 
                                                                                    self.receiver_stress12] if receiver])

    def record_stress(self, ti, fieldBuffer):
        self.record_stress11(ti, fieldBuffer.stress11)
        self.record_stress22(ti, fieldBuffer.stress22)
        self.record_stress12(ti, fieldBuffer.stress12)

    def record_velocity(self, ti, fieldBuffer):
        self.record_vel1(ti, fieldBuffer.vel1)
        self.record_vel2(ti, fieldBuffer.vel2)
    
    def draw_on_map(self, map, type):
        receiver = getattr(self, type)
        self._transducer_manager.draw_on_map( receiver, map)
        
    def has_no_receivers(self):
        return self._transducer_manager.has_no_transducers()
    
    @property
    def num_receivers(self):
        return self._transducer_manager.num_transducers
        
    @property
    def receivers(self):
        return self._transducer_manager.transducers
    
    @property
    def time_unit(self):
        return self._transducer_manager.time_unit
        
    @property
    def delta(self):
        return self._transducer_manager.delta
        
    @property
    def offset(self):
        return self._transducer_manager.offset
            
    def record_vel1(self, time: float, field: np.ndarray) -> None:
        self.receiver_vel1.record_signal(time, field)

    def record_vel2(self, time: float, field: np.ndarray) -> None:
        self.receiver_vel2.record_signal(time, field)

    def record_stress11(self, time: float, field: np.ndarray) -> None:
        self.receiver_stress11.record_signal(time, field)

    def record_stress22(self, time: float, field: np.ndarray) -> None:
        self.receiver_stress22.record_signal(time, field)

    def record_stress12(self, time: float, field: np.ndarray) -> None:
        self.receiver_stress12.record_signal(time, field)
                
class EmitterSet2D:
    
    def __init__(self, **kwargs):
        self.emitter_vel1 : Emitter = kwargs.pop('emitter_vel1', [])
        self.emitter_vel2 : Emitter = kwargs.pop('emitter_vel2', [])
        self.emitter_stress11 : Emitter = kwargs.pop('emitter_stress11', [])
        self.emitter_stress22 : Emitter = kwargs.pop('emitter_stress22', [])
        self.emitter_stress12 : Emitter = kwargs.pop('emitter_stress12', [])
        
        self._transducer_manager = TransducerManager( [emitter for emitter in [self.emitter_vel1, 
                                                                               self.emitter_vel2, 
                                                                               self.emitter_stress11, 
                                                                               self.emitter_stress22, 
                                                                               self.emitter_stress12] if emitter])
        
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
        self._transducer_manager.draw_on_map( emitter, map);
        
    def has_no_emitters(self):
        return self._transducer_manager.has_no_transducers()
        
    @property
    def emitters(self):
        return self._transducer_manager.transducers
                
    @property
    def time_unit(self):
        return self._transducer_manager.time_unit
    
    @property
    def delta(self):
        return self._transducer_manager.delta
        
    @property
    def offset(self):
        return self._transducer_manager.offset
    
    @property
    def num_emitters(self):
        return self._transducer_manager.num_transducers
                
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
    
class TransducerManager:
    def __init__(self, transducers):
        self.transducers = transducers
        self._validate_transducers()
        
    def _validate_transducers(self):
        transducer = self.transducers
        nDevices = len(transducer)
        
        for i in range(1, nDevices):
            sig = transducer[i].signal
            if sig.dims[0].delta != self.delta or sig.dims[0].offset != self.offset or sig.dims[0].quantity.si_unit != self.time_unit:
                raise ValueError("All signals must match in delta, offset, and si_unit")

    def draw_on_map(self, transducer, map):
        if transducer:
            pos = transducer.position
            nPos = pos.shape[0]
            # iterate over all positions
            for pi in range(nPos):
                map[pos[pi, 0], pos[pi, 1]] = 1
    
    def has_no_transducers(self):
        return len(self.transducers) == 0
      
    @property            
    def num_transducers(self):
        return len( self.transducers)

    @property
    def time_unit(self):
        transducer = self.transducers
        if transducer:
            return transducer[0].signal.dims[0].quantity.si_unit
        else:
            return ""
        
    @property
    def delta(self):
        transducer = self.transducers
        if transducer:
            return transducer[0].signal.dims[0].delta
        else:
            return 1
        
    @property
    def offset(self):
        transducer = self.transducers
        if transducer:
            return transducer[0].signal.dims[0].offset
        else:
            return 0
        
           
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