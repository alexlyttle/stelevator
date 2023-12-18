import numpy as np
import pandas as pd
from numpy import array, nan
from stelevator.emulators import Emulator, MESASolarLikeEmulator


class TestEmulator:
    inputs = []
    outputs = []
    summary = (
          'Emulator\n'
        + '========\n\n'
        + 'Inputs\n'
        + '------\n\n'
        + 'Outputs\n'
        + '-------\n'
    )

    def test_init(self):
        """Test Emulator class initialization."""
        emulator = Emulator(self.inputs, self.outputs)
        assert emulator.inputs == self.inputs
        assert emulator.outputs == self.outputs
    
    def test_summary(self):
        """Test Emulator class summary."""
        emulator = Emulator(self.inputs, self.outputs)
        assert emulator.summary == self.summary


class TestMESASolarLikeEmulator:
    emulator = MESASolarLikeEmulator()
    x = [0.5, 1.0, 2.0, 0.28, 0.0181]
    y = [6.35464806e-01, 5.71249586e+03, 1.00409413e+00, 1.35017031e+02, 9.92808460e-02]
    x_out = [2.5, 1.0, 2.0, 0.28, 0.0181]
    grid_inputs = dict(
        f_evol=np.linspace(0.0, 2.0, 5),
        mass=1.0,
        a_MLT=2.0,
        initial_Y=0.28,
        initial_Z=0.0181,
    )
    grid_numpy = array(
        [
            [nan, nan, nan, nan, nan],
            [6.35464806e-01, 5.71249586e+03, 1.00409413e+00, 1.35017031e+02, 9.92808460e-02],
            [9.36956554e-01, 5.71525434e+03, 1.22708526e+00, 1.00467365e+02, 6.55477519e-02],
            [1.00787014e+00, 5.64153625e+03, 1.41402101e+00, 8.16422869e+01, 8.23972573e-02],
            [nan, nan, nan, nan, nan]
        ]
    )
    grid_columns = pd.Index(['log_age', 'Teff', 'radius', 'delta_nu', 'surface_M_H'], dtype='object')
    grid_index = pd.MultiIndex.from_tuples(
        [
            (0.0, 1.0, 2.0, 0.28, 0.0181),
            (0.5, 1.0, 2.0, 0.28, 0.0181),
            (1.0, 1.0, 2.0, 0.28, 0.0181),
            (1.5, 1.0, 2.0, 0.28, 0.0181),
            (2.0, 1.0, 2.0, 0.28, 0.0181)
        ],
        names=['f_evol', 'mass', 'a_MLT', 'initial_Y', 'initial_Z']
    )

    def test_validate(self):
        assert not self.emulator.validate(self.x_out)

    def test_model(self):
        """Test MESASolarLikeEmulator.model()."""
        assert np.isclose(self.emulator(self.x), self.y).all()

    def test_grid(self):
        grid_outputs = self.emulator.grid(**self.grid_inputs)
        assert np.isclose(grid_outputs.to_numpy(), self.grid_numpy, equal_nan=True).all()
        assert grid_outputs.columns.equals(self.grid_columns)
        assert grid_outputs.index.equals(self.grid_index)

    def test_call(self):
        assert np.isclose(self.emulator(self.x), self.y).all()

    def test_call_invalid(self):
        assert np.isnan(self.emulator(self.x_out)).all()
