import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt
from typing import Tuple
from numpy.typing import NDArray
import numpy.linalg as la
from scipy.optimize import minimize


class Experiment:
    def __init__(self, *args: int, **kwargs) -> None:
        self.RNG = np.random.default_rng(1337)
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
            self.n_neurons = args[0]
            self.n_trials = args[1]
            size = (self.n_neurons, self.n_trials)
            self._stimuli = 2 * self.RNG.normal(mu, sigma, size)
        else:
            self.n_trials = () if args == None else args[0]
            self.n_neurons = 1
            self._stimuli = 2*self.RNG.normal(mu, sigma, (self.n_trials))

    def get_tuning_curve(self, N: int):
        return sig.windows.gaussian(N, 5) * np.cos(2*np.pi*np.arange(N)/10)

    def activity(self):
        tcurve = self.get_tuning_curve(self.n_trials)
        self.intensity = np.exp(self._stimuli @ tcurve)
        return self.intensity


class Neuron(Experiment):
    def __init__(self, *args: int, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def spiking(self):
        if self.n_trials > 1:
            return self.RNG.poisson(lam=self.activity(), size=(self.n_trials,))
        else:
            return self.RNG.poisson(lam=self.activity())


########################### OBJECTIVE FUNCTIONS ###########################

def objPriorRegularized(g, X, r, sig):
    lam = np.exp(X @ g)
    return -r.T @ np.log(lam) + lam.sum() + (g.T @ la.inv(sig) @ g) + np.sum(la.inv(sig) @ g)


def objPriorUnregularized(g, X, r, sig):
    lam = np.exp(X @ g)
    return -r.T @ np.log(lam) + lam.sum() + (g.T @ la.inv(sig) @ g)


def objMLL(g, X, r):
    lam = np.exp(X @ g)
    return -r.T @ np.log(lam) + lam.sum()


########################### OPTIMIZATION MAIN FUN ###########################
opt = {'disp': True,
       'maxiter': 1000,
       'gtol': 1e-9}


def optimization(obj: callable,
                 n_neurons: int = 100,
                 n_trials: int = 100,
                 random_init: bool = False,
                 stimulus_scale: float = 1.0,
                 *, opt: dict = opt
                 ) -> NDArray:

    neurons = Neuron(n_neurons, n_trials, mu=0, sigma=1)
    true_g = neurons.get_tuning_curve(neurons.n_trials)

    MAP_FUNCS = [
        'objPriorUnregularized',
        'objPriorRegularized',
    ]

    ML_FUNCS = [
        'objMLL'
    ]

    # Parameters to be passed to objective function
    resp_m = neurons.activity()
    stimuli_m = stimulus_scale * neurons.stimuli

    if obj.__name__ in MAP_FUNCS:
        sig_scale = 1
        sig = sig_scale * np.eye(true_g.shape[0])
        args = (stimuli_m, resp_m, sig)
    elif obj.__name__ in ML_FUNCS:
        sig = None
        args = (stimuli_m, resp_m)
    else:
        raise ValueError(f'Objective function {obj.__name__} not recognized.')

    if random_init:
        g_init = np.random.randn(true_g.shape[0])
    else:
        g_init = np.zeros_like(true_g)

    resultMAP = minimize(fun=obj,
                         x0=g_init,
                         args=args,
                         method='BFGS',
                         options=opt)
    return resultMAP.x
