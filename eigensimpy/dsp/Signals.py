import numpy as np

class Quantity:
    """
    A class that represents a physical or non-physical quantity, such as length, mass, time, etc.

    Attributes:
        _name (str): The name of the quantity.
        _si_unit (str): The SI unit of the quantity.
    """
    
    def __init__(self, **kwargs):
        self._name = kwargs.get('Name', '')
        self._si_unit = kwargs.get('SiUnit', '')
        
    @property
    def Label(self):
        """
        Returns a string that represents the name of the quantity, along with its SI unit (if any), enclosed in square brackets.

        Returns:
            str: The label of the quantity.
        """
        
        label = self._name
        if self._si_unit != '':
            label += f' [{self._si_unit}]'
        return label
    
    def __eq__(self, other):
        return (self._si_unit == other._si_unit) and (self._name == other._name)
    
    def __ne__(self, other):
        return not self.__eq__(other)
    


class Dimension:
    def __init__(self, **kwargs):
        self.Delta = kwargs.get('Delta', 1)
        self.Offset = kwargs.get('Offset', 0)
        self.Quantity = kwargs.get('Quantity', Quantity())
        
    @property
    def SiUnit(self):
        return self.Quantity.SiUnit
    
    @SiUnit.setter
    def SiUnit(self, value):
        self.Quantity.SiUnit = value
        
    @property
    def Name(self):
        return self.Quantity.Name
    
    @Name.setter
    def Name(self, value):
        self.Quantity.Name = value
        
    @property
    def Label(self):
        return self.Quantity.Label
    
    def __eq__(self, other):
        return (self.Delta == other.Delta) and (self.Offset == other.Offset) and (self.Quantity == other.Quantity)
    
    def __ne__(self, other):
        return not self.__eq__(other)
    
    def dimVector(self, numElem):
        basevec = np.arange(numElem)
        prod = self.Delta * basevec
        vec = self.Offset + prod
        return vec
    
    def findDimName(self, dimName):
        if dimName == ":":
            return np.ones(len(self), dtype=bool)
        else:
            dimensionNames = np.array([d.Name for d in self])
            return dimensionNames == dimName
        
    def findDimNameIndex(self, dimName):
        matchingDims = self.findDimName(dimName)
        return np.flatnonzero(matchingDims)
    
    def setDimUnit(self, dim, dimUnit):
        dimNum = self._determineDimInput(dim)
        self._mustMatchInputSize(dimNum, dimUnit)
        for i in dimNum:
            self[i].Quantity.SiUnit = dimUnit
    
    def setDimName(self, dim, name):
        dimNum = self._determineDimInput(dim)
        self._mustMatchInputSize(dimNum, name)
        for i in dimNum:
            self[i].Quantity.Name = name
    
    def setDimDelta(self, dim, delta):
        dimNum = self._determineDimInput(dim)
        self._mustMatchInputSize(dimNum, delta)
        for i in dimNum:
            self[i].Delta = delta
    
    def setDimOffset(self, dim, offset):
        dimNum = self._determineDimInput(dim)
        self._mustMatchInputSize(dimNum, offset)
        for i in dimNum:
            self[i].Offset = offset
    
    def _determineDimInput(self, dimName):
        if isinstance(dimName, str):
            dimNum = self.findDimNameIndex(dimName)
        elif isinstance(dimName, (int, float)):
            dimNum = [dimName]
        elif isinstance(dimName, bool):
            dimNum = np.flatnonzero(dimName)
        else:
            raise ValueError('Dimension index must be numeric or a string')
        return dimNum
    
    def _mustMatchInputSize(self, index, var):
        if len(index) == 1 and np.size(var) != 1:
            raise ValueError('Number of arguments does not match number of found or specified dimensions')
        elif np.size(var) not in (1, len(index)):
            raise ValueError('Number of arguments does not match number of found or specified dimensions')