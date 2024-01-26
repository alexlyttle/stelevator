import os, h5py
import numpy as np
import pandas as pd
import astropy.units as u
from warnings import warn
from numpy.typing import ArrayLike
from collections import namedtuple
from .parameters import Parameter
from .utils import _DATADIR


class Emulator(object):
    """Base class for emulators.
    
    Args:
        inputs (ParameterList): Input parameters for emulator.
        outputs (ParameterList): Output parameters for emulator.
    """
    def __init__(self, inputs: list[Parameter], outputs: list[Parameter]):
        self._inputs = inputs
        self._outputs = outputs
        self._summary = self._make_summary()

    def _make_summary(self) -> str:
        name = self.__class__.__name__
        summary = (
            f'{name}\n'
            + '='*len(name)
            + '\n\nInputs\n------\n'
            + '\n'.join([f'{i}.' for i in self.inputs])
            + '\n\nOutputs\n-------\n'
            + '\n'.join([f'{o}.' for o in self.outputs])
        )
        return summary

    @property
    def inputs(self) -> list[Parameter]:
        return self._inputs

    @property
    def outputs(self) -> list[Parameter]:
        return self._outputs

    @property
    def summary(self) -> str:
        return self._summary

    def model(self, x: ArrayLike) -> np.ndarray:
        """Returns the model output for the given input.

        This returns the raw model output, without input validation.

        Args:
            x (ArrayLike): Input to the model.

        Returns:
            ArrayLike: Model output.
        """
        raise NotImplementedError(f"Model for '{self.__class__.__name__}' is not yet implemented.")

    def grid(self, **inputs) -> pd.DataFrame:
        """Returns a grid of model outputs for the product of the given inputs.
        
        Args:
            **inputs: Keyword arguments for the inputs to the model.
        
        Returns:
            pd.DataFrame: Table with grid inputs as the index and outputs as the columns.
        """
        dimensions = [np.atleast_1d(inputs.pop(i.name)).ravel() for i in self.inputs]

        if len(inputs) > 0:
            warn(f"Unknown keyword arguments have been ignored: {', '.join(inputs.keys())}.")
        
        input_names = [i.name for i in self.inputs]
        index = pd.MultiIndex.from_product(dimensions, names=input_names)

        output_names = [o.name for o in self.outputs]
        df = pd.DataFrame(index=index, columns=output_names).reset_index()
        
        df[output_names] = self(df[input_names])
        return df.set_index(input_names)

    def error(self, x: np.ndarray) -> np.ndarray:
        """Return estimate of the error at a given input. This is the truth minus the model output.

        The estimate comes from the emulators test dataset. This could output a distribution or
        parameters for a distribution.

        If the mean error is zero then its variance can just be added to that of the likelihood
        during inference.
        """
        raise NotImplementedError(f"Error for '{self.__class__.__name__}' is not yet implemented.") 

    def validate(self, x: np.ndarray) -> bool:
        raise NotImplementedError(f"Validation for '{self.__class__.__name__}' is not yet implemented.")

    def __call__(self, x: ArrayLike) -> ArrayLike:
        """Returns the model output for the given input.
        
        Inputs are validated against the domain of the emulator. 

        Args:
            x (ArrayLike): Input to the model.

        Returns:
            ArrayLike: Model output.
        """
        x = np.asarray(x)
        if x.shape[-1] != len(self.inputs):
            raise ValueError(f"Input must have {len(self.inputs)} dimensions.")
        mask = self.validate(x)
        y = self.model(x)
        # TODO: alternatively, warn or raise error if any inputs are outside the domain
        y[~mask] = np.nan
        return y


class MESASolarLikeEmulator(Emulator):
    """Emulator for the MESA solar-like oscillator model from Lyttle et al. (2021).
    
    This emulator was trained on data in the range:
    
    0.01 <= f_evol < 2.00
    0.8 <= M/Msun < 1.2
    1.5 <= alpha_MLT < 2.5
    0.22 <= Y < 0.32
    0.005 <= Z < 0.04
    """

    _filename = 'lyttle21.weights.h5'

    def __init__(self):
        inputs = [
            Parameter('f_evol', 'f_\\mathrm{evol}', desc='Fractional evolutionary phase'),
            Parameter('mass', 'M', 'Msun', desc='Stellar mass'),
            Parameter('a_MLT', r'\alpha_\mathrm{MLT}', desc='Mixing length parameter'),
            Parameter('initial_Y', 'Y_\\mathrm{init}', desc='Initial stellar helium mass fraction'),
            Parameter('initial_Z', 'Z_\\mathrm{init}', desc='Initial stellar heavy element mass fraction'),
        ]
        outputs = [
            Parameter('log_age', '\\log_{10}(t)', 'dex', desc='Logarithm of stellar age in Gyr'),
            Parameter('Teff', 'T_\\mathrm{eff}', 'K', desc='Stellar effective temperature'),
            Parameter('radius', 'R', 'Rsun', desc='Stellar radius'),
            Parameter('delta_nu', r'\Delta\nu', 'uHz', desc='Asteroseismic large frequency separation'),
            Parameter('surface_M_H', r'[\mathrm{M}/\mathrm{H}]_\mathrm{surf}', 'dex', desc='Surface metallicity')
        ]
        super().__init__(inputs, outputs)

        self.loc = (
            np.array([0.865, 1.0, 1.9, 0.28, 0.017]),
            np.array([0.79, 5566.772, 1.224, 100.72, 0.081])
        )
        self.scale = (
            np.array([0.651, 0.118, 0.338, 0.028, 0.011]),
            np.array([0.467, 601.172, 0.503, 42.582, 0.361])
        )
        self.bounds = (
            np.array([0.01, 0.8, 1.5, 0.22, 0.005]),
            np.array([2.0, 1.2, 2.5, 0.32, 0.04])
        )
        self.weights, self.bias = self._load_weights()

    def _load_weights(self):
        """Loads the model weights and biases from file.
        
        Returns:
            tuple: Tuple of lists containing the weights and biases for each layer.
        """
        filename = os.path.join(_DATADIR, self._filename)
        with h5py.File(filename, 'r') as file:
            weights = [file['dense']['dense']['kernel:0'][()]]
            bias = [file['dense']['dense']['bias:0'][()]]
            for i in range(1, 7):
                weights.append(file[f'dense_{i}'][f'dense_{i}']['kernel:0'][()])
                bias.append(file[f'dense_{i}'][f'dense_{i}']['bias:0'][()])
        return weights, bias

    def validate(self, x: np.ndarray) -> bool:
        """Validates the input against the domain of the emulator.
        
        Returns:
            bool: True if all inputs are in the domain, False otherwise.
        """
        return np.all((x >= self.bounds[0]) & (x <= self.bounds[1]), axis=-1)

    def model(self, x):
        x = (x - self.loc[0]) / self.scale[0]
        for w, b in zip(self.weights[:-1], self.bias[:-1]):
            x = np.dot(x, w) + b
            x = np.where(x >= 0, x, (np.exp(x) - 1))
        x = np.dot(x, self.weights[-1]) + self.bias[-1]
        return self.loc[1] + self.scale[1] * x


class MESADeltaScutiEmulator(Emulator):
    """Emulator for the MESA δ Sct oscillator model from Scutt et al. (in review)."""


class MESARotationEmulator(Emulator):
    """Emulator for the MESA rotation model from Saunders et al. (in preparation)."""


class YRECRotationEmulator(Emulator):
    """Emulator for the YREC rotation model from Saunders et al. (in preparation)."""
