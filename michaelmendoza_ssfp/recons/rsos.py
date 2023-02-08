import numpy as np

def rsos(input):
    ''' Computes root sum of squares '''
    return np.sqrt(np.sum(np.square(np.abs(input)), axis = 3))