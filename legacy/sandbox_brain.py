import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import normalized_root_mse
from skimage.restoration import unwrap_phase
import random
from tqdm import tqdm
from time import perf_counter

from phantom import load_dataslice, load_dataset, generate_offres, generate_phantom, get_phantom_parameters
from mri_ssfp import ma_ssfp, add_noise_gaussian
from ssfp import bssfp, planet
from ormtre import ormtre

def brain_example():
    N = 128
    npcs = 6 

    filepath = './data'
    data = load_dataslice(filepath, image_index=1, slice_index=150)

    freq = 500
    offres = generate_offres(N, f=freq, rotate=True, deform=True) 

    # alpha = flip angle
    alpha = np.deg2rad(60)

    #Create brain phantom
    phantom = generate_phantom(data, alpha, offres=offres)

    #Get phantom parameter
    M0, T1, T2, _alpha, df, _sample = get_phantom_parameters(phantom)

    # Generate phase-cycled images 
    TR = 3e-3
    TE = TR / 2
    pcs = np.linspace(0, 2 * np.pi, npcs, endpoint=False)
    M = ma_ssfp(T1, T2, TR, TE, alpha, f0=df, dphi=pcs, M0=M0)
    M = add_noise_gaussian(M, sigma=0.015)

    print(M.shape)

    # Show the phase-cycled images
    nx, ny = 2, 3
    plt.figure()
    for ii in range(nx*ny):
        plt.subplot(nx, ny, ii+1)
        plt.imshow(np.abs(M[:, :, ii]))
        plt.title('%d deg PC' % (ii*(360/npcs)))
    plt.show()

def brain_planet_example():
    N = 128; npcs = 8 
    filepath = './data'
    data = load_dataslice(filepath, image_index=1, slice_index=150)

    freq = 1 / 3e-3
    offres = generate_offres(N, f=freq, rotate=True, deform=True) 

    # alpha = flip angle
    alpha = np.deg2rad(15)

    #Create brain phantom
    phantom = generate_phantom(data, alpha, offres=offres)

    #Get phantom parameter
    M0, T1, T2, _alpha, df, _sample = get_phantom_parameters(phantom)

    # Generate phase-cycled images 
    TR = 3e-3
    TE = TR / 2
    pcs = np.linspace(0, 2 * np.pi, npcs, endpoint=False)
    M = ma_ssfp(T1, T2, TR, TE, alpha, f0=-df, dphi=pcs, M0=M0)
    #M = add_noise_gaussian(M, sigma=0.0)
    M = np.transpose(M, (2,0,1))
    M = M[...,None]

    # Do T1, T2 mapping for each pixel
    mask = np.abs(M0) > 1e-8
    mask = mask[...,None]

    print(M.shape, alpha, TR, mask.shape)

    # Show the phase-cycled images
    nx, ny = 2, 4
    plt.figure()
    for ii in range(nx*ny):
        plt.subplot(nx, ny, ii+1)
        plt.imshow(np.abs(M[ii, :, :, 0]))
        plt.title('%d deg PC' % (ii*(360/npcs)))
    plt.show()

    # Do the thing
    t0 = perf_counter() 
    Mmap, T1est, T2est, dfest = planet(M, alpha, TR, mask=mask, pc_axis=0)
    print('Took %g sec to run PLANET' % (perf_counter() - t0))

   # Look at a single slice

    print(T1est.shape, T2est.shape, dfest.shape, T1.shape, T2.shape, mask.shape)
    sl = 0
    T1est = T1est[..., sl]
    T2est = T2est[..., sl]
    dfest = dfest[..., sl]
    #T1 = T1[..., sl]
    #T2 = T2[..., sl]
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
    
def brain_planet_virtual_ellipse_example():
    N = 128; 
    filepath = './data'
    data = load_dataslice(filepath, image_index=1, slice_index=150)

    freq = 1 / 3e-3
    offres = generate_offres(N, f=freq, rotate=True, deform=True) 

    # alpha = flip angle
    alpha = np.deg2rad(100)

    #Create brain phantom
    phantom = generate_phantom(data, alpha, offres=offres)

    #Get phantom parameter
    M0, T1, T2, _alpha, df, _sample = get_phantom_parameters(phantom)

    # Generate phase-cycled images 
    TR = 3e-3; TE = TR / 2
    pcs = np.linspace(0, 2 * np.pi, 4, endpoint=False)
    M = ma_ssfp(T1, T2, TR, TE, alpha, f0=-df, dphi=pcs, M0=M0)
    #M = add_noise_gaussian(M, sigma=0.0)
    M = np.transpose(M, (2,0,1))
    M = M[...,None]

    TR1 = 3.45e-3; TE1 = TR / 2
    pcs = np.linspace(0, 2 * np.pi, 4, endpoint=False)
    M1 = ma_ssfp(T1, T2, TR1, TE1, alpha, f0=-df, dphi=pcs, M0=M0)
    M1 = np.transpose(M1, (2,0,1))
    M1 = M1[...,None]

    # Do T1, T2 mapping for each pixel
    mask = np.abs(M0) > 1e-8
    mask = mask[...,None]
    print(M.shape, alpha, TR, mask.shape)

    # Look at a single slice
    sl = 0
    mask = mask[..., sl]

    print(M.shape, M1.shape, M.shape[-1], M1.shape[-1])
    phi = ormtre(M, M1, mask, TR, TR1, pc_axis=0, rad=True)

    plt.figure()
    plt.imshow(phi)
    plt.show()

    print('M, phi', M.shape, phi[..., None].shape)
    v = np.empty(M.shape, dtype=M.dtype)
    v[..., :4] = M
    v[..., 4:] = M1*np.exp(1j*phi[None, ...])

    print('v', v.shape)

    t0 = perf_counter() 
    Mmap, T1est, T2est, dfest = planet(v, alpha=alpha, TR=(TR + TR1)/2, mask=mask, pc_axis=0)
    print('Took %g sec to run PLANET' % (perf_counter() - t0))

   # Look at a single slice
    sl = 0
    T1est = T1est[..., sl]
    T2est = T2est[..., sl]
    dfest = dfest[..., sl]
    #T1 = T1[..., sl]
    #T2 = T2[..., sl]
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

def brain_dataset_example():
    N = 128
    npcs = 8
    freq = 500

    data = load_dataset('./data')
    dataset = []

    for i in tqdm(range(data.shape[0])):

        # Generate off resonance 
        offres = generate_offres(N, f=freq, rotate=True, deform=True) 

        # alpha = flip angle
        alpha = np.deg2rad(60)

        # Create brain phantom
        phantom = generate_phantom(data, alpha, img_no=i, offres=offres)

        # Get phantom parameter
        M0, T1, T2, _alpha, df, _sample = get_phantom_parameters(phantom)

        # Generate phase-cycled images 
        TR = 3e-3
        TE = TR / 2
        pcs = np.linspace(0, 2 * np.pi, npcs, endpoint=False)
        M = ma_ssfp(T1, T2, TR, TE, alpha, f0=df, dphi=pcs, M0=M0)
        M = add_noise_gaussian(M, sigma=0.015)
        dataset.append(M[None, ...])
    
    dataset = np.concatenate(dataset, axis=0)
    print(dataset.shape)
    
    # Show the phase-cycled images
    nx, ny = 2, 4
    plt.figure()
    for ii in range(nx*ny):
        plt.subplot(nx, ny, ii+1)
        plt.imshow(np.abs(dataset[0,:, :, ii]))
        plt.title('%d deg PC' % (ii*(360/npcs)))
    plt.show()

#brain_example()
#brain_dataset_example()
#brain_planet_example()
brain_planet_virtual_ellipse_example()
