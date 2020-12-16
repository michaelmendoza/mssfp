import numpy as np
import matplotlib.pyplot as plt
import random
from tqdm import tqdm

from phantom import data_loader, dataset_loader, generate_offres, generate_phantom, get_phantom_parameters
from mri_ssfp import ma_ssfp, add_noise_gaussian

def brain_example():
    N = 128
    npcs = 6 

    filepath = './data'
    data = data_loader(filepath, image_count=1, slice_index=150)

    freq = 500
    offres = generate_offres(N, f=freq, rotate=True, deform=True) 

    # alpha = flip angle
    alpha = np.deg2rad(60)

    #Create brain phantom
    phantom = generate_phantom(data, alpha, offres=offres)

    #Get phantom parameter
    M0, T1, T2, alpha, df, _sample = get_phantom_parameters(phantom)

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

def brain_dataset_example():
    N = 128
    npcs = 8
    freq = 500

    data = dataset_loader('./data')
    dataset = []

    for i in tqdm(range(data.shape[0])):

        # Generate off resonance 
        offres = generate_offres(N, f=freq, rotate=True, deform=True) 

        # alpha = flip angle
        alpha = np.deg2rad(60)

        # Create brain phantom
        phantom = generate_phantom(data, alpha, img_no=i, offres=offres)

        # Get phantom parameter
        M0, T1, T2, alpha, df, _sample = get_phantom_parameters(phantom)

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
        plt.imshow(np.abs(dataset[100,:, :, ii]))
        plt.title('%d deg PC' % (ii*(360/npcs)))
    plt.show()

#brain_example()
brain_dataset_example()
