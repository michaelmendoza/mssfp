from typing import Tuple, Any
import math
from tqdm import tqdm
import numpy as np
from ..simulations import ssfp, add_noise_gaussian

# T1/T2 values taken from https://mri-q.com/why-is-t1--t2.html
tissue_map = {
    0: ('none', 0, 0, 0),
    1: ('csf', 4.2, 1.99, 0),
    2: ('gray-matter', 0.9, 0.1, 0, 0),
    3: ('white-matter', 0.6, 0.08, 0, 0),
    4: ('muscle', 0.9, .05, 0),
    5: ('liver', 0.5, 0.04, 0), 
    6: ('fat', 0.25, 0.07, 420),   # 420Hz is the average fat signal at 3T - from chemical shift
    7: ('tendon', 0.4, 0.005, 0),
    8: ('proteins', 0.250, 0.001, 0)
}

def get_relaxation_times(tissue_id: int) -> Tuple[float, float, float]:
    """Get T1 and T2 relaxation times for a given tissue ID."""
    _, t1, t2, f0 = tissue_map[tissue_id]
    return t1, t2, f0

def generate_block_phantom2d(shape: int=256, padding: int=32):
    s = (shape - 2 * padding, shape - 2 * padding)
    block = np.ones(s) * 1.0
    block = np.pad(block, (padding, padding))
    block.astype(int)
    return block

def generate_blocks_phantom2d(shape: int=256, blocks: int=8, padding: int=8):
    width = 64
    height = 64

    s = (width - 2 * padding, height - 2 * padding)
    patches = [[], []]
    for i in range(blocks+1):
        if i > 0:
            patch = np.ones(s) * i
            patch = np.pad(patch, (padding, padding))
            patches[int((i - 1) / 4)].append(patch)
        
    mask = np.block(patches)
    mask = mask.astype(int)
    mask = np.pad(mask, ((64, 64), (0, 0)))

    return mask

def generate_circle_phantom2d(h, w, center=None, radius=None):

    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask

def generate_circles_phantom2d(shape: int= 256, count: int = 9, padding: int =8):
    width = 256
    height = 256
    mask = np.zeros((width, height))
    pass

def generate_line_phantom2d(length: int=256, padding: int=32):
    s = (1, length) 
    line = np.ones(s) * 1.0

    if padding > 0:
        line[0, -1] = 0
        line[0, -padding:-1] = 0
        line[0, 0:padding] = 0
        
    line.astype(int)
    return line 

def generate_ssfp_phantom(phantom_type = 'block', slices = 128, TR = 3e-3, TE = 3e-3 / 2, alpha = np.deg2rad(15), npcs = 4, sigma = 0.005):
    
    generate_phantom = {
        'block': generate_block_phantom2d,
        'blocks': generate_blocks_phantom2d,
        'circle': generate_circle_phantom2d,
        'circles': generate_circles_phantom2d, 
        'line': generate_line_phantom2d,
    }

    phantom = generate_phantom[phantom_type]()
    shape = phantom.shape
    mask = (phantom != 0) * 1

    M0 = mask
    T1 = np.zeros((slices, shape[0], shape[1]))
    T2 = np.zeros((slices, shape[0], shape[1]))
    F0 = np.zeros((slices, shape[0], shape[1]))
    for tissue_id in tissue_map:
            mask = (phantom == tissue_id)
            t1, t2, f0 = get_relaxation_times(tissue_id)
            T1[mask] = t1
            T2[mask] = t2
            F0[mask] = f0
        
    # Simulation SSFP with phantom data 
    dataset = []
    pcs = np.linspace(0, 2 * math.pi, npcs, endpoint=False)
    for i in tqdm(range(slices)):
        M = ssfp(T1, T2, TR, TE, alpha, f0=F0, dphi=pcs, M0=M0)
        M = add_noise_gaussian(M, sigma=sigma)
        dataset.append(M[None, ...])

    dataset = np.concatenate(dataset, axis=0)
    return { 'M': dataset, 'phantom': phantom }