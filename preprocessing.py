import pandas as pd
import numpy as np
from numba import njit, prange


def get_cutpoint(data, num_bins=10, method='EWB'):
    if method == 'EWB':
        return pd.cut(data, bins=num_bins, retbins=True)[1]

    elif method == 'EFB':
        return pd.qcut(data, q=num_bins, retbins=True)[1]

def transition_matrix(array1, array2, cut_points):
    # Bin the arrays using pandas' cut() function
    bins1 = pd.cut(array1, bins=cut_points, labels=False, include_lowest=True)
    bins2 = pd.cut(array2, bins=cut_points, labels=False, include_lowest=True)

    # Calculate the transition matrix
    transition_matrix, _, _ = np.histogram2d(bins1, bins2, bins=(len(cut_points)-1, len(cut_points)-1))

    return transition_matrix


#normalize matrix

def normalize(matrix):
    #return matrix / matrix.sum(axis=1, keepdims=True)
    return matrix / np.sum(matrix)


# Average sample

def avg_sample(df, day, window_size = 10):
    if day < window_size:
        return np.mean(df[df.columns[:-1]].loc[:window_size-1], axis = 0).astype(int)
    else:
        return np.mean(df[df.columns[:-1]].loc[day-window_size-1:day-1], axis = 0).astype(int)


# Average transition matrix

def avg_transmatrix(df, day, cut_points, window_size=10):
    transmatrixes = []
    non_event = []

    if day < window_size:
        for i in range(window_size - 1):
            if df['class'][i] == 0:
                non_event.append(i)
    else:
        for i in range(day - window_size, day - 1):
            if df['class'][i] == 0:
                non_event.append(i)
    for i in range(len(non_event) - 1):
        transmatrixes.append(
            transition_matrix(df[df.columns[:-1]].loc[non_event[i]], df[df.columns[:-1]].loc[non_event[i + 1]],
                              cut_points=cut_points))

    transmatrixes = np.array(transmatrixes)

    return np.mean(transmatrixes, axis=0).astype(int)

# transition matrix of sample on a day and average sample

def transmatrix(df, day, cut_points, window_size = 10):
    return transition_matrix(df[df.columns[:-1]].loc[day], avg_sample(df, day, window_size), cut_points)