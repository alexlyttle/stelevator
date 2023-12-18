import astropy.units as u
from stelevator.parameters import Parameter

name = 'time'
symbol = 't'
unit = 's'
desc = 'Time parameter'

def test_parameter_init():
    """Test Parameter class initialization."""
    p = Parameter(name, symbol=symbol, unit=unit, desc=desc)
    assert p.name == name
    assert p.symbol == symbol
    assert p.unit == u.Unit(unit)
    assert p.desc == desc

def test_parameter_init_no_symbol():
    """Test Parameter class initialization without symbol."""
    p = Parameter(name, unit=unit, desc=desc)
    assert p.name == name
    assert p.symbol == r'\mathrm{' + name + '}'
    assert p.unit == u.Unit(unit)
    assert p.desc == desc

def test_parameter_repr():
    """Test Parameter class representation."""
    p = Parameter(name, symbol=symbol, unit=unit, desc=desc)
    assert repr(p) == f"Parameter({name!r}, symbol={symbol!r}, unit={u.Unit(unit).to_string()!r}, desc={desc!r})"
