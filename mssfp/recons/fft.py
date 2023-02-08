import numpy as np

def fft2d(data, doInverse = False, axis = (1,2)):
    ''' Calculates 2D FFT for numpy data. Set doInverse = True to do ifft.  '''
    x, y = (axis)

    if (doInverse):
        fft = np.fft.ifft
        shift = np.fft.ifftshift
        unshift = np.fft.fftshift
        A = 1 * np.sqrt(data.shape[x]* data.shape[y]) 
    else:
        fft = np.fft.fft
        shift = np.fft.fftshift
        unshift = np.fft.ifftshift
        A = 1 / np.sqrt(data.shape[x]* data.shape[y]) 

    result = shift(shift(data, axes=(x,)), axes=(y,))
    result = fft(fft(result, axis = x), axis = y)
    result = unshift(unshift(result, axes=(x,)), axes=(y,))
    return A * result 

def ifft2d(data, axis=(1,2)):
    ''' Calculates 2D iFFT for numpy data. '''
    return fft2d(data, True, axis)