import numpy as np
import matplotlib.pyplot as plt

class Signal:
    def __init__(self,**kwargs):
        
        empty_array = np.zeros((0, 0), dtype=int);
        self._data: np.array = kwargs.get('data', empty_array)
        self._dims: DimensionArray = kwargs.get('dims', DimensionArray())
        self._amplitude: Quantity = kwargs.get('amplitude', Quantity())
        
        if len(self._dims.dim_array) < self.ndim:
            raise ValueError("Dimension array must have the same number of elements as the data shape")
                
    def copy(self):
        copied_data = self._data.copy()
        copied_dims = self._dims.copy()
        copied_amplitude = self._amplitude.copy()
        return Signal(data=copied_data, dims=copied_dims, amplitude=copied_amplitude)
      
    def plot(self, ax=None):
        
        N = self.data.shape[0]
        x_data = self.dims[0].dim_vector( N )
        y_data = self.data

        if ax is None:
            fig, ax = plt.subplots()
            # show_plot = True
        # else:
        #     show_plot = False

        ax.plot(x_data, y_data)

        ax.set_xlabel(self.dims[0].label)
        ax.set_ylabel(self.amplitude.label)
        ax.set_title('Signal Data')

        # if show_plot:
        #     plt.show()

        return ax
                
    @property
    def ndim(self):
        return self._data.ndim
    
    @property
    def amplitude(self):
        return self._amplitude
        
    @property
    def shape(self):
        return self._data.shape
        
    @property
    def data(self):
        return self._data
        
    @property
    def dims(self):
        return self._dims
    
    def is_view(self):
        if self.data.base is None:
            return False
        else:
            return True
        
    
    def crop(self, start, end):
        
        N = self.shape[0]
        start_idx = self.dims[0].to_sample(start)
        end_idx   = self.dims[0].to_sample(end)
        
        if start_idx < 0:
            raise ValueError("Start index is out of bounds")
            
        if end_idx >= N:
            raise ValueError("End index is out of bounds")
                
        self._data = self._data[start_idx:end_idx]
        self._dims[0].offset = start;        
        
        return self
        
    def permute(self, order):
        
        self._data = np.transpose(self._data, order)
        self._dims = [self._dims[i] for i in order]
        
        return self
    
    
    def __repr__(self):
        return f"Signal(data={self._data}, dims={self._dims.dim_array})"
    
class Quantity:
    """
    A class that represents a physical or non-physical quantity, such as length, mass, time, etc.

    Attributes:
        _name (str): The name of the quantity.
        _si_unit (str): The SI unit of the quantity.
    """
    
    def __init__(self, **kwargs):
        self._name = kwargs.get('name', '')
        self._si_unit = kwargs.get('si_unit', '')
        
    @property
    def name(self):
        return self._name
    
    @property
    def si_unit(self):
        return self._si_unit
    
    @name.setter
    def name(self, value):
        self._name = value
    
    @si_unit.setter
    def si_unit(self, value):
        self._si_unit = value
            
    @property
    def label(self):
        """
        Returns a string that represents the name of the quantity, along with its SI unit (if any), enclosed in square brackets.

        Returns:
            str: The label of the quantity.
        """
        
        label = self.name
        if self.si_unit != '':
            label += f' [{self.si_unit}]'
        return label
    
    def copy(self):
        return Quantity(name=self.name, si_unit=self.si_unit)
    
    
    def __eq__(self, other):
        return (self.si_unit == other.si_unit) and (self.name == other.name)
    
    def __ne__(self, other):
        return not self.__eq__(other)
    

class Dimension:
    
    def __init__(self, **kwargs):
        self._delta: np.float64 = kwargs.get('delta', 1.0)
        self._offset: np.float64 = kwargs.get('offset', 0.0)
        self.quantity: Quantity = kwargs.get('quantity', Quantity())
        
        if not isinstance(self._delta, np.float64):
            self._delta = np.float64(self._delta)

        if not isinstance(self._offset, np.float64):
            self._offset = np.float64(self._offset)

    def copy(self):
        return Dimension(delta=self.delta, offset=self.offset, quantity=self.quantity.copy())
    
    def to_sample(self, value):
        """ converts the given value into unit values of this dimension"""
        return int( round((value - self._offset) / self._delta ))
    
    def to_unitvalue(self, sample):
        """ converts the given value into unit values of this dimension"""
        return self._offset + np.float64(sample) * self._delta
        
    @property
    def delta(self) -> np.float64:
        return self._delta

    @delta.setter
    def delta(self, value):
        if not isinstance(value, np.float64):
            value = np.float64(value)
        self._delta = value

    @property
    def offset(self) -> np.float64:
        return self._offset

    @offset.setter
    def offset(self, value):
        if not isinstance(value, np.float64):
            value = np.float64(value)
        self._offset = value        
        
    @property
    def si_unit(self):
        return self.quantity.si_unit
    
    @si_unit.setter
    def si_unit(self, value):
        self.quantity.si_unit = value
        
    @property
    def name(self):
        return self.quantity.name
    
    @name.setter
    def name(self, value):
        self.quantity.name = value
        
    @property
    def label(self):
        return self.quantity.label
    
    def __eq__(self, other):
        return (self.delta == other.delta) and (self.offset == other.offset) and (self.quantity == other.quantity)
    
    def __ne__(self, other):
        return not self.__eq__(other)
    
    def dim_vector(self, numElem):
        basevec = np.arange(numElem).astype(np.float64)
        prod = self.delta * basevec
        vec = self.offset + prod
        return vec
    

class DimensionArray:
    def __init__(self, *args):
        
        if not args:  # Empty input
            self.dim_array = np.array([], dtype=Dimension)  
        elif len(args) == 1:
            arg = args[0]
            if isinstance(arg, Dimension):  # Single Dimension input
                self.dim_array = np.array([arg], dtype=Dimension)
            elif isinstance(arg, (list, tuple, np.ndarray)) and all(isinstance(dim, Dimension) for dim in arg): # Array of Dimension input
                self.dim_array = np.array(arg, dtype=Dimension) 
            else:
                raise ValueError("Invalid input. Please provide an empty input, a single Dimension object, or an array of Dimension objects.")
        else:
            raise ValueError("Invalid input. Please provide an empty input, a single Dimension object, or an array of Dimension objects.")
        
    def find_dim_name(self, dimName):        
        return np.array([d.name == dimName for d in self.dim_array])

    def find_dim_nameIndex(self, dimName):
        matchingDims = self.find_dim_name(dimName)
        return np.flatnonzero(matchingDims)
    
    def __getitem__(self, index):
        return self.dim_array[index]
    
    def __setitem__(self, index, dimension):
        self.dim_array[index] = dimension
    
    def __len__(self):
        return len(self.dim_array)
    
    def copy(self):
        copied_dim_array = [dim.copy() for dim in self.dim_array]
        return DimensionArray(copied_dim_array)
    