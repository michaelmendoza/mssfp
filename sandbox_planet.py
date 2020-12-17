'''Show basic usage of GS solution.'''

from time import perf_counter

import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import normalized_root_mse
from skimage.restoration import unwrap_phase
from phantominator import shepp_logan
from ssfp import bssfp, planet

from glob import glob
from os.path import isfile
from time import time

def load_data():
    filename = "C:/Users/michael.mendoza/projects/ssfp/mrdata/20190507_GASP_LONG_TR_WATER/set1_tr24_te12.npy"
    data = np.load(filename)
    #data = np.mean(data, axis=2) # Average coils
    data = np.mean(data, axis=3) # Average averages
    data = data[:,:,:,0::2]

    filename = "C:/Users/michael.mendoza/projects/ssfp/mrdata/20190507_GASP_LONG_TR_WATER/gre_field_mapping.npy"
    data2 = np.load(filename)
    #data2 = np.mean(data2, axis=2) # Average coils

    print(data.shape)
    print(data2.shape)

    # Show the phase-cycled images
    nx, ny = 2, 4
    plt.figure()
    for ii in range(nx*ny):
        plt.subplot(nx, ny, ii+1)
        plt.imshow(np.abs(data[:, :, 0, ii]))
    plt.show() 

    # Show field map
    df = np.squeeze(data2[:,:,:,1] - data2[:,:,:,0])
    plt.imshow(np.abs(df[:,:,0]))
    plt.show()

    return data, df

def planet_phantom_example():

    TR, alpha = 24e-3, np.deg2rad(70)

    sig, df = load_data()
    coil_index = 0
    sig = sig[:,:,coil_index,:]
    df = df[:,:,coil_index]
    
    sig = sig.transpose((2, 0, 1)) # Move pc to 0 index
    sig = sig[..., None]

    # Do T1, T2 mapping for each pixel
    mask = np.abs(sig[1,:,:,:]) > 5e-8

    print('-------')
    print(sig.shape)
    print(df.shape)
    print(mask.shape)
    
    # Do the thing
    t0 = perf_counter()
    Mmap, T1est, T2est, dfest = planet(sig, alpha, TR, pc_axis=0)
    print('Took %g sec to run PLANET' % (perf_counter() - t0))

    print(sig.shape)
    print(df.shape)
    print(dfest.shape)

    # Simple phase unwrapping of off-resonance estimate
    dfest = unwrap_phase(dfest*2*np.pi*TR)/(2*np.pi*TR)

    nx, ny = 3, 3

    plt.subplot(nx, ny, 2)
    plt.imshow(T1est)
    plt.title('T1 est')
    plt.axis('off')

    plt.subplot(nx, ny, 5)
    plt.imshow(T2est)
    plt.title('T2 est')
    plt.axis('off')

    plt.subplot(nx, ny, 7)
    plt.imshow(np.abs(df))
    plt.title('df Truth')
    plt.axis('off')

    plt.subplot(nx, ny, 8)
    plt.imshow(np.abs(dfest))
    plt.title('df est')
    plt.axis('off')

    plt.subplot(nx, ny, 9)
    plt.imshow(np.abs(df - dfest[:,:,0]))
    plt.title('NRMSE: %g' % normalized_root_mse(df, dfest[:,:,0]))
    plt.axis('off')

    plt.show()

def planet_shepp_logan_example():

    # Shepp-Logan
    N, nslices, npcs = 128, 1, 8  # 2 slices just to show we can
    M0, T1, T2 = shepp_logan((N, N, nslices), MR=True, zlims=(-.25, 0))

    # Simulate bSSFP acquisition with linear off-resonance
    TR, alpha = 3e-3, np.deg2rad(15)
    pcs = np.linspace(0, 2*np.pi, npcs, endpoint=False)
    df, _ = np.meshgrid(
        np.linspace(-1/TR, 1/TR, N),
        np.linspace(-1/TR, 1/TR, N))
    sig = np.empty((npcs,) + T1.shape, dtype='complex')
    for sl in range(nslices):
        sig[..., sl] = bssfp(
            T1[..., sl], T2[..., sl], TR, alpha, field_map=df,
            phase_cyc=pcs, M0=M0[..., sl])

    # Do T1, T2 mapping for each pixel
    mask = np.abs(M0) > 1e-8

    # Make it noisy
    np.random.seed(0)
    sig += 1e-5*(np.random.normal(0, 1, sig.shape) +
                 1j*np.random.normal(0, 1, sig.shape))*mask

    print(sig.shape, alpha, TR, mask.shape)
    
   # Show the phase-cycled images
    nx, ny = 2, 4
    plt.figure()
    for ii in range(nx*ny):
        plt.subplot(nx, ny, ii+1)
        plt.imshow(np.abs(sig[ii, :, :, 0]))
        plt.title('%d deg PC' % (ii*(360/npcs)))
    plt.show()

    # Do the thing
    t0 = perf_counter()
    Mmap, T1est, T2est, dfest = planet(sig, alpha, TR, mask=mask, pc_axis=0)
    print('Took %g sec to run PLANET' % (perf_counter() - t0))

    print(T1est.shape, T2est.shape, dfest.shape, T1.shape, T2.shape, mask.shape)

    # Look at a single slice
    sl = 0
    T1est = T1est[..., sl]
    T2est = T2est[..., sl]
    dfest = dfest[..., sl]
    T1 = T1[..., sl]
    T2 = T2[..., sl]
    mask = mask[..., sl]

    # Simple phase unwrapping of off-resonance estimate
    dfest = unwrap_phase(dfest*2*np.pi*TR)/(2*np.pi*TR)

    print('t1, mask:', T1.shape, mask.shape)

    nx, ny = 3, 3
    plt.subplot(nx, ny, 1)
    plt.imshow(T1*mask)
    plt.title('T1 Truth')
    plt.axis('off')

    plt.subplot(nx, ny, 2)
    plt.imshow(T1est)
    plt.title('T1 est')
    plt.axis('off')

    plt.subplot(nx, ny, 3)
    plt.imshow(T1*mask - T1est)
    plt.title('NRMSE: %g' % normalized_root_mse(T1, T1est))
    plt.axis('off')

    plt.subplot(nx, ny, 4)
    plt.imshow(T2*mask)
    plt.title('T2 Truth')
    plt.axis('off')

    plt.subplot(nx, ny, 5)
    plt.imshow(T2est)
    plt.title('T2 est')
    plt.axis('off')

    plt.subplot(nx, ny, 6)
    plt.imshow(T2*mask - T2est)
    plt.title('NRMSE: %g' % normalized_root_mse(T2, T2est))
    plt.axis('off')

    plt.subplot(nx, ny, 7)
    plt.imshow(df*mask)
    plt.title('df Truth')
    plt.axis('off')

    plt.subplot(nx, ny, 8)
    plt.imshow(dfest)
    plt.title('df est')
    plt.axis('off')

    plt.subplot(nx, ny, 9)
    plt.imshow(df*mask - dfest)
    plt.title('NRMSE: %g' % normalized_root_mse(df*mask, dfest))
    plt.axis('off')

    plt.show()


if __name__ == '__main__':
    #load_data()
    planet_shepp_logan_example()
    #planet_phantom_example()