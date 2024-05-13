from scipy.io import loadmat
from typing import Tuple
from numpy.typing import NDArray


def loadData(idx, fpath) -> Tuple[NDArray, NDArray]:
    # load a dictionary
    data = loadmat(fpath)
    data = data['Data']
    # Unpack the tuple
    dat, labels = data[:, idx].item()
    return dat.T, labels


def get_delayed_matrices(mat):
    # beginning changes in the system
    x0 = mat[:, :-1]
    x1 = mat[:, 1:]
    return x0, x1


def get_expected_dynamics(mat, A):

    m = mat.shape[1]
    new_dyanmics = []

    for i in range(m):
        xt = A * mat[:, i]
        new_dyanmics.append(xt)

    return new_dyanmics
