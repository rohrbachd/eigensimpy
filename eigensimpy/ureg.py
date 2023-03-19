from pint import UnitRegistry

unit = UnitRegistry()

def neper(value):
    return unit.Quantity(value, 'neper')