import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt
import seaborn as sns
import polars as pl
from numpy.typing import NDArray

RNG = np.random.default_rng(1337)


class Experiment:
    def __init__(self, *args: int, **kwargs) -> None:
        self.stimuli = (args, kwargs)
        # self.kwargs = kwargs

    @property
    def stimuli(self):
        return self._stimuli

    @stimuli.setter
    def stimuli(self, values):
        args, kwargs = values

        if len(kwargs) > 1:
            mu = kwargs.get('mu', 0)
            sigma = kwargs.get('sigma', 1)
        else:
            mu = 0
            sigma = 1

        if len(args) > 1:
            self.n_trials = args[0]
            self.n_neurons = args[1]
            size = (self.n_trials, self.n_neurons)
            self._stimuli = 2 * RNG.normal(mu, sigma, size)
        else:
            self.n_trials = () if args == None else args[0]
            self.n_neurons = 1
            self._stimuli = 2*RNG.normal(mu, sigma, (self.n_trials))

    def get_tuning_curve(self, N: int):
        return sig.windows.gaussian(N, 5) * np.cos(2*np.pi*np.arange(N)/10)

    def activity(self):
        if self.n_neurons == 1:  # () integers
            tcurve = self.get_tuning_curve(self.n_trials)
            self.intensity = np.exp(self._stimuli @ tcurve)
            return self.intensity
        elif self.n_neurons > 1:
            tmat = np.vstack([self.get_tuning_curve(self.n_neurons)
                             for _ in range(self.n_trials)])
            self.intensity = np.exp(self._stimuli @ tmat)
            return self.intensity
        else:
            print('Invalid number of neurons')


class Neruon(Experiment):
    def __init__(self, *args: int, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def spiking(self):
        if self.n_trials > 1:
            return RNG.poisson(lam=self.activity(), size=(self.n_trials,))
        else:
            return RNG.poisson(lam=self.activity())
