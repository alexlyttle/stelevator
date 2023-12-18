import os
from stelevator.utils import _PACKAGEDIR, _DATADIR

def test_packagedir_exists():
    """Test that the package directory exists."""
    assert os.path.exists(_PACKAGEDIR)


def test_datadir_exists():
    """Test that the data directory exists."""
    assert os.path.exists(_PACKAGEDIR)
