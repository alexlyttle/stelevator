import numpy as np
import astropy.units as u
from .utils import _ListSameType


class Parameter(object):
    """Base class for parameters.

    Args:
        name (str): Name of the parameter.
        symbol (str): LaTeX symbol of the parameter. Defaults to None.
        unit (str or astropy.units.Unit): Unit of the parameter. Defaults to None.
        desc (str): Description of the parameter. Defaults to None.
    """
    def __init__(self, name, symbol=None, unit=None, desc=None):
        self.name = name
        self.unit = u.Unit('' if unit is None else unit)
        self.symbol = r'\mathrm{' + name + '}' if symbol is None else symbol
        self.desc = name if desc is None else desc

    def __repr__(self):
        return f"{self.__class__.__name__}('{self.name}', symbol='{self.symbol}', unit='{self.unit.to_string()}', desc='{self.desc}')"
