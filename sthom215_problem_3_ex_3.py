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
