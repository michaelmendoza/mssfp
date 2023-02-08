import numpy as np
import matplotlib.pyplot as plt
from skimage import io, transform
from .fft import fft2d, ifft2d
from .rsos import rsos

def super_fov(data, ref = None, doUndersample = False, doPlot = False):
    ''' Computes superFOV recon. Note: Expects data in k-space.'''

    if(doUndersample):
        _ = undersample(data)
        data = _[0]
        ref = _[1]

    data = ifft2d(data)
    ref = ifft2d(ref)
    smap = sensitivity_map(ref)
    smap_ = resize_complex(smap, data.shape)
    recon = SENSE(data[0], smap_[0], 2)

    if doPlot:
        plt.figure()
        plt.imshow(np.abs(recon), vmax=0.3)
        plt.title('Reconstructed Image')
        plt.xticks([])
        plt.yticks([])
        plt.show()

    return recon

def acs(kspace, shape):
    ''' Samples the center lines of k-space for auto-calibrating singal (ACS) region. '''
    ny, nx = kspace.shape[1:3]
    [cny, cnx] = shape
    idxx = int(np.floor((nx - cnx) / 2))
    idxy = int(np.floor((ny - cny) / 2))
    result = kspace[:, idxy:idxy+cny, idxx:idxx+cnx, ...]
    return result

def undersample(data, type = 'SENSE', undersampling_ratio = 2, doFFT = True):
    ''' Undersampled data. Can only undersampled by integer values. '''
    if doFFT:
        data = fft2d(data)

    if type == "SENSE":
        ref = acs(data, (32,32))
        undersampling_ratio = int(undersampling_ratio)    
        mask = np.zeros(data.shape)
        mask[:, ::undersampling_ratio, ...] = 1
    
    data = data * mask
    return [data, ref]

def sensitivity_map(data, axis = 3):
    ''' Generates a sensitivity map. Expects 4-d data (slice, height, width, channel) '''
    results = np.zeros(data.shape, dtype= complex)
    image = rsos(data)
    for i in range(data.shape[axis]):
        results[:,:,:,i] = data[:,:,:,i] / image
    return results

def SENSE(data, sensitivity_map, R):
    ''' SENSE Reconstruction for Parallel Imaging '''
    [height, width, _] = sensitivity_map.shape
    results = np.zeros([height, width], dtype= complex)
    
    for y in range(int(height / R)):
        index = np.arange(y, height, int(height / R))
        for x in range(width):
            s = np.transpose(sensitivity_map[index, x, :].reshape(R, -1))
            M = np.matmul(np.linalg.pinv(s), data[y, x, :].reshape(-1, 1))    
            results[index, x] = M[:,0]
    return results

def resize_complex(data, size):
    ''' Resize complex image to match a certain size '''
    data_real = transform.resize(np.real(data), size, mode='constant')
    data_img = transform.resize(np.imag(data), size, mode='constant')
    return data_real + 1j*data_img