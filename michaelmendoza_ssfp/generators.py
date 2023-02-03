from typing import Any
import math
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage import io, transform

from .mri_ssfp import ma_ssfp, add_noise_gaussian

# T1/T2 values taken from https://mri-q.com/why-is-t1--t2.html
tissue_map = {
    'none': [0, 0],
    'water': [4, 2],
    'white-matter': [0.6, 0.08],
    'gray-matter': [0.9, 0.1],
    'muscle': [0.9, .05],
    'liver': [0.5, 0.04],
    'fat': [0.25, 0.07],
    'tendon': [0.4, 0.005],
    'proteins': [0.250, 0.001]
}

def generate_shepp_phantom2d(height, width):
        filename = 'shepp256.png'
        img = mpimg.imread(filename)
        img = np.array(img)
        img = transform.resize(img, (height, width), mode='constant')

def generate_block_phantom2d(padding = 8, f0 = 4 / 3e-3):
    width = 64
    height = 64
    keys = list(tissue_map.keys())

    s = (width - 2 * padding, height - 2 * padding)
    patches = [[],[]]
    for i in range(len(keys)):
        if i > 0:
            patch = np.ones(s) * i
            patch = np.pad(patch, (padding, padding))
            patches[int((i - 1) / 4)].append(patch)
        
    mask : Any = np.block(patches)
    mask = mask.astype(int)
    size = mask.shape

    M0 = (mask != 0) * 1

    t1 = list(map(lambda x: tissue_map[keys[x]][0], mask.flatten()))
    t1 = np.array(t1).reshape(mask.shape)

    t2 = list(map(lambda x: tissue_map[keys[x]][1], mask.flatten()))
    t2 = np.array(t2).reshape(mask.shape)

    f = np.linspace(-f0, f0, size[1])
    f = np.tile(f, (size[0], 1))

    phantom = {'mask':mask, 'M0':M0, 't1':t1, 't2':t2, 'fo':f}
    return phantom

def generate_circle_phantom2d():
    width = 256
    height = 256
    mask = np.zeros((width, height))
    pass

def create_circular_mask(h, w, center=None, radius=None):

    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask

def generate_block_phantom2d_ssfp(depth = 128, TR = 3e-3, TE = 3e-3 / 2, alpha = np.deg2rad(15), npcs = 4, sigma = 0.005):
    
    phantom = generate_block_phantom2d()
    mask = phantom['mask']
    M0 = phantom['M0']
    T1 = phantom['t1']
    T2 = phantom['t2']
    df = phantom['fo']
        
    # Simulation SSFP with phantom data 
    dataset = []
    pcs = np.linspace(0, 2 * math.pi, npcs, endpoint=False)
    for i in tqdm(range(depth)):
        M = ma_ssfp(T1, T2, TR, TE, alpha, field_map=df, dphi=pcs, M0=M0)
        M = add_noise_gaussian(M, sigma=sigma)
        dataset.append(M[None, ...])

    dataset = np.concatenate(dataset, axis=0)
    return { 'M': dataset, 'phantom': phantom }