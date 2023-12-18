import astropy.units as u


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

    def _format_unit(self) -> str:
        unit = self.unit.to_string()
        return f' ({unit})' if unit != '' else ''

    def __repr__(self):
        return f"{self.__class__.__name__}('{self.name}', symbol='{self.symbol}', unit='{self.unit.to_string()}', desc='{self.desc}')"

    def __str__(self):
        unit = self._format_unit()
        return f'{self.name}{unit}: {self.desc}'
