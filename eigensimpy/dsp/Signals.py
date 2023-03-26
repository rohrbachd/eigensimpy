import numpy as np
import matplotlib.pyplot as plt
  

class Signal:
    def __init__(self,**kwargs):
        
        empty_array = np.zeros((0, 0), dtype=int)
        
        data = kwargs.get('data', empty_array)
        if not isinstance(data, np.ndarray):
            self._data: np.array = np.array(data)
        else:
            self._data: np.array = data

        # Check if dims is a Dimension or a list/tuple of Dimensions
        dims = kwargs.get('dims', [])
        if isinstance(dims, Dimension):
            dims = [dims]

        # If the provided dimensions are less than the data dimensions, fill with default Dimensions
        while len(dims) < self._data.ndim:
            dims.append(Dimension())

        self._dims: DimensionArray = DimensionArray(dims)

        # Check if an Amplitude Quantity object is provided
        if isinstance(kwargs.get('amplitude'), Quantity):
            self._amplitude: Quantity = kwargs.get('amplitude')
        else:
            # Create an Amplitude Quantity object using the provided amp_name and amp_si_unit
            self._amplitude: Quantity = Quantity(name=kwargs.get('amp_name', 'Amplitude'), 
                                                 si_unit=kwargs.get('amp_si_unit', ''))
                

    def num_signals(self, axis):
        """Calculate the number of signals along the given axis."""
        
        flattened_shape = np.delete(self.shape, axis)
        return np.prod(flattened_shape)

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

        lines = ax.plot(x_data, y_data)

        ax.set_xlabel(self.dims[0].label)
        ax.set_ylabel(self.amplitude.label)
        ax.set_title('Signal Data')

        # if show_plot:
        #     plt.show()

        return ax, lines
    
    def imshow(self, ax=None):
        if len(self.data.shape) < 2:
            raise ValueError("Data must be a 2D array for imshow.")

        if ax is None:
            fig, ax = plt.subplots()

        # in images and figures the first dimension (0) is y, Vertical
        # the second dimension (1) is x, Horizontal
        y_data = self.dims[0].dim_vector(self.data.shape[0])
        x_data = self.dims[1].dim_vector(self.data.shape[1])

        img = ax.imshow(self.data, origin='lower', extent=[x_data[0], x_data[-1], y_data[0], y_data[-1]], aspect='auto')

        ax.set_ylabel(self.dims[0].label)
        ax.set_xlabel(self.dims[1].label)
        ax.set_title('Signal Data as Image')

        plt.colorbar(img, ax=ax, label=self.amplitude.label)

        return ax, img
                
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
        
    def value_at(self, position_unit, axis=0) -> np.ndarray:
        
        N = self.shape[axis]
        sample_idx = self.dims[axis].to_sample(position_unit)
        
        if sample_idx < 0:
            raise ValueError("Start index is out of bounds")
            
        if sample_idx >= N:
            raise ValueError("End index is out of bounds")
        # will create ndim slice objects. Each slice object has 3 elements
        # (start, stop, step)        
        slices = [slice(None)] * self.ndim
        slices[axis] = slice(sample_idx, sample_idx+1)
                
        data = self._data[tuple(slices)]
        
        return data
        
    def set_value_at(self, value, position_unit, axis=0) -> None:
        
        N = self.shape[axis]
        sample_idx = self.dims[axis].to_sample(position_unit)
        
        if sample_idx < 0:
            raise ValueError("Start index is out of bounds")
            
        if sample_idx >= N:
            raise ValueError("End index is out of bounds")
            
        # will create ndim slice objects. Each slice object has 3 elements
        # (start, stop, step) 
        slices = [slice(None)] * self.ndim
        slices[axis] = slice(sample_idx, sample_idx+1)
                
        self._data[tuple(slices)] = value       
        
                
    def crop(self, start, end, axis=0):
        
        N = self.shape[axis]
        start_idx = self.dims[axis].to_sample(start)
        end_idx   = self.dims[axis].to_sample(end)
        
        if start_idx < 0:
            raise ValueError("Start index is out of bounds")
            
        if end_idx >= N:
            raise ValueError("End index is out of bounds")
        
        corrected_start = self.dims[axis].to_unitvalue(start_idx)         
        # will create ndim slice objects. Each slice object has 3 elements
        # (start, stop, step)        
        slices = [slice(None)] * self.ndim
        slices[axis] = slice(start_idx, end_idx)
                
        self._data = self._data[tuple(slices)]
        self._dims[axis] = self._dims[axis].new( offset = corrected_start );        
        
        return self
        
    def permute(self, order): 
        
        self._data = np.transpose(self._data, order)
        self._dims = [self._dims[i] for i in order]
        
        return self
    
    """ ***** Overwrite [] indexing """    
    def __getitem__(self, index):
        return self._data[index]

    def __setitem__(self, index, value):
        self._data[index] = value
    
    def __repr__(self):
        return f"Signal(data={self._data}, dims={self._dims.dim_array})"
    
    """ ***** Basic Signal Class operations """
    def __add__(self, other):
        if isinstance(other, Signal):
            result_data = self._data + other._data
        else:
            result_data = self._data + other
        return Signal(data=result_data, dims=self._dims.copy(), amplitude=self._amplitude)

    def __sub__(self, other):
        if isinstance(other, Signal):
            result_data = self._data - other._data
        else:
            result_data = self._data - other
        return Signal(data=result_data, dims=self._dims.copy(), amplitude=self._amplitude)

    def __mul__(self, other):
        if isinstance(other, Signal):
            result_data = self._data * other._data
        else:
            result_data = self._data * other
        return Signal(data=result_data, dims=self._dims.copy(), amplitude=self._amplitude)

    def __truediv__(self, other):
        if isinstance(other, Signal):
            result_data = self._data / other._data
        else:
            result_data = self._data / other
        return Signal(data=result_data, dims=self._dims.copy(), amplitude=self._amplitude)

    def __matmul__(self, other):
        if isinstance(other, Signal):
            result_data = self._data @ other._data
        else:
            result_data = self._data @ other
        return Signal(data=result_data, dims=self._dims.copy(), amplitude=self._amplitude)

    def __pow__(self, other):
        result_data = self._data ** other
        return Signal(data=result_data, dims=self._dims.copy(), amplitude=self._amplitude)

    def __radd__(self, other):
        return self.__add__(other)

    def __rsub__(self, other):
        return self.__neg__().__add__(other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __rtruediv__(self, other):
        if isinstance(other, np.ndarray):
            result_data = other / self._data
        else:
            result_data = np.divide(other, self._data)
        return Signal(data=result_data, dims=self._dims.copy(), amplitude=self._amplitude)

    def __rmatmul__(self, other):
        result_data = other @ self._data
        return Signal(data=result_data, dims=self._dims.copy(), amplitude=self._amplitude)
    
    def __neg__(self):
        neg_data = -self._data
        return Signal(data=neg_data, dims=self._dims.copy(), amplitude=self._amplitude.copy())


class Quantity:
    """
    A class that represents a physical or non-physical quantity, such as length, mass, time, etc.

    Attributes:
        _name (str): The name of the quantity.
        _si_unit (str): The SI unit of the quantity.
        
    Quantity is immutable. use the new method to create variants of an object    
    """
    
    def __init__(self, **kwargs):
        
        self._is_mutable = True
        self._name = kwargs.get('name', '')
        self._si_unit = kwargs.get('si_unit', '')
        self._is_mutable = False
    
    def new(self, **kwargs):
        new_kwargs = {
            'name': kwargs.get('name', self.name),
            'si_unit': kwargs.get('si_unit', self.si_unit)
        }
        return Quantity(**new_kwargs)    
    
    @property
    def name(self):
        return self._name
    
    @property
    def si_unit(self):
        return self._si_unit
    
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


    def __setattr__(self, name, value):
        if hasattr(self, '_is_mutable') and not self._is_mutable:
            raise AttributeError("Cannot modify an immutable Quantity instance")
        super().__setattr__(name, value)
        
    def __str__(self):
        return f"Quantity(name='{self.name}', si_unit='{self.si_unit}')"

    def __eq__(self, other):
        return (self.si_unit == other.si_unit) and (self.name == other.name)
    
    def __ne__(self, other):
        return not self.__eq__(other)

            
            
class Dimension:
    
    def __init__(self, **kwargs):
        
        self._is_mutable = True
        self._delta: np.float64 = kwargs.get('delta', 1.0)
        self._offset: np.float64 = kwargs.get('offset', 0.0)
        
        # Check if a Quantity object is provided
        if isinstance(kwargs.get('quantity'), Quantity):
            self.quantity: Quantity = kwargs.get('quantity')
        else:
            # Create a Quantity object using the provided name and si_unit
            self.quantity: Quantity = Quantity(name = kwargs.get('name', 'Samples'), 
                                               si_unit = kwargs.get('si_unit', ''))
        
        if not isinstance(self._delta, np.float64):
            self._delta = np.float64(self._delta)

        if not isinstance(self._offset, np.float64):
            self._offset = np.float64(self._offset)

        self._is_mutable = False
    
    def new(self, **kwargs):
        
        if isinstance(kwargs.get('quantity'), Quantity):
            new_kwargs = {
                'delta': kwargs.get('delta', self._delta),
                'offset': kwargs.get('offset', self._offset),
                'quantity': kwargs.get('quantity', self.quantity)
            }
        else:
            new_kwargs = {
                'delta': kwargs.get('delta', self._delta),
                'offset': kwargs.get('offset', self._offset),
                'name': kwargs.get('name', self.quantity.name),
                'si_unit': kwargs.get('si_unit', self.quantity.si_unit)
            }
        return Dimension(**new_kwargs)

    def __str__(self):
        return f"Dimension(delta={self._delta}, offset={self._offset}, quantity={self.quantity})"

    
    def dim_vector(self, numElem):
        basevec = np.arange(numElem).astype(np.float64)
        prod = self.delta * basevec
        vec = self.offset + prod
        return vec    

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

    @property
    def offset(self) -> np.float64:
        return self._offset
                   
    @property
    def si_unit(self):
        return self.quantity.si_unit
    
    @property
    def name(self):
        return self.quantity.name
        
    @property
    def label(self):
        return self.quantity.label
    
    def __eq__(self, other):
        return (self.delta == other.delta) and (self.offset == other.offset) and (self.quantity == other.quantity)
    
    def __ne__(self, other):
        return not self.__eq__(other)
    
    def __setattr__(self, name, value):
        if hasattr(self, '_is_mutable') and not self._is_mutable:
            raise AttributeError("Cannot modify an immutable Dimension instance")
        super().__setattr__(name, value)
        
        
    # @si_unit.setter
    # def si_unit(self, value):
    #     self.quantity.si_unit = value
            
    # @delta.setter
    # def delta(self, value):
    #     if not isinstance(value, np.float64):
    #         value = np.float64(value)
    #     self._delta = value
        
    # @offset.setter
    # def offset(self, value):
    #     if not isinstance(value, np.float64):
    #         value = np.float64(value)
    #     self._offset = value        
    
    # @name.setter
    # def name(self, value):
    #     self.quantity.name = value    
    

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
            elif isinstance(arg, DimensionArray) and all(isinstance(dim, Dimension) for dim in arg):
                self.dim_array = np.array(arg.dim_array, dtype=Dimension)   
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
    
    def __eq__(self, other):
        if isinstance(other, DimensionArray):
            if len(self) == len(other):
                return all(dim1 == dim2 for dim1, dim2 in zip(self.dim_array, other.dim_array))
        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    
    def copy(self):
        copied_dim_array = [dim for dim in self.dim_array]
        return DimensionArray(copied_dim_array)
    