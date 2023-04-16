from abc import ABC, abstractmethod
import matplotlib.pyplot as plt

class AFieldRecorder(ABC):

    @abstractmethod
    def record(self, field):
        pass

    @abstractmethod
    def initialize(self, field):
        pass


class RecorderSet2D:

    def __init__(self, **kwargs):
        self.recorder_vel1: RecorderList = self._ensure_list(kwargs.pop('recorder_vel1', []))
        self.recorder_vel2: RecorderList = self._ensure_list(kwargs.pop('recorder_vel2', []))
        self.recorder_stress11: RecorderList = self._ensure_list(kwargs.pop('recorder_stress11', []))
        self.recorder_stress22: RecorderList = self._ensure_list(kwargs.pop('recorder_stress22', []))
        self.recorder_stress12: RecorderList = self._ensure_list(kwargs.pop('recorder_stress12', []))

    @staticmethod
    def _ensure_list(item):
        if isinstance(item, AFieldRecorder):
            return RecorderList([item])
        elif isinstance(item, list):
            return RecorderList(item)
        elif isinstance(item, RecorderList):
            return item
        else:
            raise ValueError("Expecting an AFieldRecorder instance or a list of AFieldRecorder instances")

    def initialize(self, field):
        success = []
        for recorder_list in [self.recorder_vel1, self.recorder_vel2, self.recorder_stress11, self.recorder_stress22, self.recorder_stress12]:
            success.append(recorder_list.initialize(field))
        return success
    
    def record_vel1(self, field):
        self.recorder_vel1.record(field)

    def record_vel2(self, field):
        self.recorder_vel2.record(field)

    def record_stress11(self, field):
        self.recorder_stress11.record(field)

    def record_stress22(self, field):
        self.recorder_stress22.record(field)

    def record_stress12(self, field):
        self.recorder_stress12.record(field)
            
class RecorderList:

    def __init__(self, recorders=None):
        
        if recorders:
            for recorder in recorders:
                if not isinstance(recorder, AFieldRecorder):
                    raise ValueError("All recorders must be instances of AFieldRecorder or its subclasses")
                
        self.recorders = recorders if recorders else []

    def add_recorder(self, recorder):
        if not isinstance(recorder, AFieldRecorder):
            raise ValueError("Recorder must be an instance of AFieldRecorder or its subclass")
        self.recorders.append(recorder)

    def initialize(self, field):
        success = []
        for recorder in self.recorders:
            success.append(recorder.initialize(field))
        return success

    def add_recorder(self, recorder):
        if not isinstance(recorder, AFieldRecorder):
            raise ValueError("Recorder must be an instance of AFieldRecorder or its subclass")
        self.recorders.append(recorder)
        
    def record(self, field):
        success = []
        for recorder in self.recorders:
            success.append(recorder.record(field))
        return success
    
    def __getitem__(self, index):
        return self.recorders[index]

    def __setitem__(self, index, recorder):
        if not isinstance(recorder, AFieldRecorder):
            raise ValueError("Recorder must be an instance of AFieldRecorder or its subclass")
        self.recorders[index] = recorder
        
    def __len__(self):
        return len(self.recorders)
    
class MessageFieldRecorder(AFieldRecorder):

    def __init__(self):
        self.messages = []

    def initialize(self, field):
        message = f"Initializing MessageFieldRecorder with field of shape {field.shape}"
        self.messages.append(message)
        return True

    def record(self, field):
        message = f"Recording field of shape {field.shape} in MessageFieldRecorder"
        self.messages.append(message)
        return True

    def get_messages(self):
        return self.messages
    
class FieldDisplay2D(AFieldRecorder):

    def __init__(self, **kwargs):
        self.fig = kwargs.get('data', None)
        self.axes = kwargs.get('axes', None)
        self.cLim = kwargs.get('cLim', None)
        self.name = kwargs.get('name', None)
        self.img = None

    def get_displayed_data(self):
    
        image = self.img
        displayed_data = None
        if image is not None:
            # displayed_data = image.norm.inverse (image.cmap( image.get_array() ) )
            displayed_data = image.get_array()
            
        return displayed_data
    
    def initialize(self, field):
        success = False
        
        if self.fig is None or not plt.fignum_exists(self.fig.number):
            self.fig = plt.figure()
        if self.axes is None or not self.axes.figure == self.fig:
            self.axes = self.fig.gca()

        self.img = self.axes.imshow(field)
        success = True
        return success

    def record(self, raw_field):
        success = False
        if self.img is None or self.img not in self.axes.images:
            self.img = self.axes.imshow(raw_field)
            success = True
        else:
            self.img.set_data(raw_field)
            if self.cLim is not None:
                self.axes.set_clim(self.cLim)
            success = True
            self.fig.canvas.draw_idle()
        return success