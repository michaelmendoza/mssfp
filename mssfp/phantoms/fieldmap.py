import numpy as np
import elasticdeform as ed
from typing import Tuple
from scipy import ndimage

def generate_fieldmap(shape: Tuple[int, ...], f: float = 300, useRotate: bool = True, 
                    useDeform: bool = True, rotation: float = 15,
                    noise_offset: float = 100, noise_sigma: float = 5) -> np.ndarray:
    """Generate off-resonance map with optional rotation and deformation."""

    # Generate linear field map
    x, _ = np.meshgrid(np.linspace(-f, f, shape[0]), np.linspace(-f, f, shape[1]))

    # Add noise
    fieldmap: np.ndarray = x 
    fieldmap += np.random.normal(noise_offset * np.random.uniform(-1, 1), noise_sigma, size=shape)
    
    if useRotate:
        rot_angle = rotation * np.random.uniform(-1, 1)
        fieldmap = ndimage.rotate(fieldmap, rot_angle, reshape=False, order=3, mode='nearest')
        
    if useDeform:
        fieldmap = ed.deform_random_grid(fieldmap, sigma=10, points=3, order=3, mode='nearest') # type: ignore
        
    return fieldmap