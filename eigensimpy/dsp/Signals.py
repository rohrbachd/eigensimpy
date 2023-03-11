import numpy as np

class Quantity:
    """
    A class that represents a physical or non-physical quantity, such as length, mass, time, etc.

    Attributes:
        _name (str): The name of the quantity.
        _si_unit (str): The SI unit of the quantity.
    """
    
    def __init__(self, **kwargs):
        self.name = kwargs.get('name', '')
        self.si_unit = kwargs.get('si_unit', '')
        
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
    
    def dimVector(self, numElem):
        basevec = np.arange(numElem).astype(np.float64)
        prod = self.delta * basevec
        vec = self.offset + prod
        return vec
    

class DimensionArray:
    def __init__(self, dim_array):
        if not all(isinstance(dim, Dimension) for dim in dim_array):
            raise ValueError("Input array must contain only Dimension objects")
        self.dim_array = np.array(dim_array)
        
    def find_dim_name(self, dimName):        
        return np.array([d.name == dimName for d in self.dim_array])

    def find_dim_nameIndex(self, dimName):
        matchingDims = self.find_dim_name(dimName)
        return np.flatnonzero(matchingDims)
    
    def set_dim_unit(self, dim, dimUnit):
        if isinstance(dim, int):
            self.dim_array[dim].quantity.si_unit = dimUnit
        else:
            for d, u in zip(dim, dimUnit):
                self.dim_array[d].quantity.si_unit = u
    
    def set_dim_name(self, dim, name):
        if isinstance(dim, int):
            self.dim_array[dim].quantity.name = name
        else:
            for d, n in zip(dim, name):
                self.dim_array[d].quantity.name = n
    
    def set_dim_delta(self, dim, delta):
        if isinstance(dim, int):
            self.dim_array[dim].delta = delta
        else:
            for d, dd in zip(dim, delta):
                self.dim_array[d].delta = dd
    
    def set_dim_offset(self, dim, offset):
        if isinstance(dim, int):
            self.dim_array[dim].offset = offset
        else:
            for d, oo in zip(dim, offset):
                self.dim_array[d].offset = oo