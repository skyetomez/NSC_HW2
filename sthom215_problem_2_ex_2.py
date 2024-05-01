from scipy.io import loadmat
import seaborn as sns
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt


def load_mat(dat: str):

    data_dict = loadmat('sample_dat')
    data = data_dict['dat']

    converted = dict()
    unique = []
    # Extract trial IDs and spike data from each tuple in the structured array
    for void_object in data:
        for thing_ple in void_object:
            trial_id = thing_ple[0].item()
            unique.append(trial_id)
            spikes = thing_ple[1]
            converted[trial_id] = spikes

    return converted


def smoothing(dat: NDArray, sigma: float, A: float) -> NDArray:
    if dat.ndim == 2:
        return _smoothing(dat, sigma, A)
    elif dat.ndim == 3:
        return _smoothing_3D(dat, sigma, A)
    else:
        raise ValueError('Input data must be 2D or 3D')


def _smoothing(dat: NDArray, sigma: float, A, ) -> NDArray:
    # dat: (num_neurons, time_bins)
    K = A * _get_kernel(dat.shape[1], sigma)
    return np.dot(dat, K)


def _smoothing_3D(tensor: NDArray, sigma: float, A: float) -> NDArray:
    # tensor shape: (trials, num_neurons, num_time_bins)
    smoothed_tensor = np.empty_like(tensor)
    for i in range(tensor.shape[0]):
        smoothed_tensor[i] = _smoothing(tensor[i], sigma, A)
    return smoothed_tensor


def _get_kernel(size: int, l: float) -> NDArray:
    K = np.zeros((size, size))
    indices = np.arange(size)
    diff = indices[:, np.newaxis] - indices
    pow = (diff / l) ** 2
    K = np.exp(-0.5 * pow)
    return K


def get_PSTH(*args, dat: NDArray) -> None:

    if len(args) == 1:
        _get_PSTH(args[0], dat)
        return None
    elif args == ():
        _get_raster(dat)
        return None

    max_ = dat.shape[-1]

    spike_data = dat[args[0], args[1], :]
    # Find indices where spikes occur
    spike_times = np.where(spike_data > 0)[0]

    # Create a figure for the raster plot
    plt.figure(figsize=(10, 2))
    plt.eventplot(spike_times, orientation='horizontal',
                  colors='black', linewidths=1.5)
    plt.title(f'Raster Plot for Neuron {args[0]} on Trial {args[1]}')
    plt.xlabel('Time Bins')
    plt.yticks([])  # No need for y-ticks in a single-trial raster
    plt.tight_layout()
    plt.show()


def _get_PSTH(trial_unit, dat) -> None:
    fig, ax = plt.subplots(figsize=(10, 10))

    # Loop through each neuron
    for neuron_index in range(dat.shape[1]):
        # Extract the spike times for the current neuron in the selected trial
        spike_data = dat[trial_unit, neuron_index, :]
        # Find indices where spikes occur
        spike_times = np.where(spike_data > 0)[0]

        # Add each neuron's raster line to the plot
        ax.eventplot(spike_times, lineoffsets=neuron_index,
                     linelengths=0.8, colors='black')

    # Formatting the plot
    ax.set_title(f'Raster Plot for All Neurons on Trial {trial_unit}')
    ax.set_xlabel('Time Bins')
    ax.set_ylabel('Neuron Index')
    ax.set_ylim(-0.5, dat.shape[1] - 0.5)
    plt.tight_layout()
    plt.show()
    return None


def _get_raster(dat):
    fig, ax = plt.subplots(figsize=(10, 10))

    # Loop through each neuron
    for neuron_index in range(dat.shape[0]):
        # Extract the spike times for the current neuron in the selected trial
        spike_data = dat[neuron_index, :]
        # Find indices where spikes occur
        spike_times = np.where(spike_data > 0)[0]

        # Add each neuron's raster line to the plot
        ax.eventplot(spike_times, lineoffsets=neuron_index,
                     linelengths=0.8, colors='black')

    # Formatting the plot
    ax.set_title(f'Raster Plot for All Neurons')
    ax.set_xlabel('Time Bins')
    ax.set_ylabel('Neuron Index')
    ax.set_ylim(-0.5, dat.shape[0] - 0.5)
    plt.tight_layout()
    plt.show()
    return None
