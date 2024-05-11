from scipy.io import loadmat
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import plotly.graph_objects as go
import seaborn as sns
from neo import SpikeTrain
import quantities as pq


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


def create_SpikeTrain(data: NDArray):

    num_trials, num_neurons, _ = data.shape
    spike_trials = list()
    spike_trains = list()

    for trial in range(num_trials):
        spikes_per_trial = list()
        for neuron in range(num_neurons):
            spike_times = np.argwhere(data[trial, neuron, :]).squeeze()
            spike_counts = data[trial, neuron, spike_times].squeeze()
            repeated_st = np.repeat(spike_times, spike_counts)
            st = SpikeTrain(repeated_st*pq.ms, units='ms', t_stop=600)
            spikes_per_trial.append(st)
        spike_trials.append(spikes_per_trial)
    return spike_trials


def pcaOverTime(mat: NDArray, num_components: int = 3):
    trials, neurons, _ = mat.shape

    transformed_mat = np.zeros((trials, neurons, num_components))

    for t in range(trials):
        data_at_time = mat[t, :, :]

        pca = PCA(n_components=num_components)
        transformed_data_at_time = pca.fit_transform(data_at_time)

        transformed_mat[t, :, :] = transformed_data_at_time.reshape(
            (1, neurons, num_components))

    return transformed_mat


def plot_principal_components(index, mat1, mat2):

    marker_size = 3

    fig1 = go.Figure(data=[go.Scatter3d(
        x=mat1[index, :, 0],
        y=mat1[index, :, 1],
        z=mat1[index, :, 2],
        mode='markers',
        marker=dict(size=marker_size)
    )])

    fig1.update_layout(title='Principal Components - Tensor 1')

    fig2 = go.Figure(data=[go.Scatter3d(
        x=mat2[index, :, 0],
        y=mat2[index, :, 1],
        z=mat2[index, :, 2],
        mode='markers',
        marker=dict(size=marker_size)
    )])

    fig2.update_layout(title='Principal Components - Tensor 2')

    fig1.update_layout(scene=dict(
        xaxis_title='X',
        yaxis_title='Y',
        zaxis_title='Z'),
        margin=dict(r=20, b=10, l=10, t=40))

    fig2.update_layout(scene=dict(
        xaxis_title='X',
        yaxis_title='Y',
        zaxis_title='Z'),
        margin=dict(r=20, b=10, l=10, t=40))

    fig1.show()
    fig2.show()


def smoothing(dat: NDArray, sigma: float, A: float) -> NDArray:
    if dat.ndim == 2:
        return _smoothing(dat, sigma, A)
    elif dat.ndim == 3:
        return _smoothing_3D(dat, sigma, A)
    else:
        raise ValueError('Input data must be 2D or 3D')


def _smoothing(dat: NDArray, sigma: float, A, ) -> NDArray:
    # dat: (num_neurons, time_bins)
    K = A * _get_kernel(dat.shape[-1], sigma)
    return np.dot(dat, K)


def _smoothing_3D(tensor: NDArray, sigma: float, A: float) -> NDArray:
    # tensor shape: (trials, num_neurons, num_time_bins)
    return np.apply_along_axis(_smoothing, 0, tensor, sigma=sigma, A=A)


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
