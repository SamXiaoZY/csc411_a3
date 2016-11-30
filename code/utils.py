import numpy as np
import pandas as pd
from scipy import misc
import glob


def Save(fname, data):
    """Saves the model to a numpy file."""
    print('Writing to ' + fname)
    np.savez_compressed(fname, **data)


def Load(fname):
    """Loads model from numpy file."""
    print('Loading from ' + fname)
    return dict(np.load(fname))


# fname = '../initial/data/*.jpg'
def LoadImages(fname):
    file_list = glob.glob(fname)
    files = np.array([np.array(misc.imread(name)) for name in file_list])
    return files


def OneHotEncode(labels, numClass=8):
    labels = labels.reshape([-1, 1])
    output = np.negative(np.ones([labels.shape[0], numClass]))
    for i, val in enumerate(labels):
        output[i][val-1] = 1
    return output
