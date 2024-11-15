import numpy as np
import elasticdeform as ed
from typing import Tuple
from scipy import ndimage

def generate_fieldmap(shape: Tuple[int, ...], f: float = 0, df: float = 300, fn_offset: float = 10, fn_sigma: float = 5,
                      rotation: float = 15, useRotate: bool = True, useDeform: bool = True) -> np.ndarray:
    """Generate off-resonance map with optional rotation and deformation.
    
    f: fieldmap offset (Hz)
    df: fieldmap range (Hz)
    fn_offset: fieldmap noise mean range (Hz)
    fn_sigma: fieldmap noise variance (Hz)
    rotation: rotation of fieldmap from (-rotation, rotation)
    useRotate: use rotation transformation
    useDeform: use deform transformation
    """

    # Generate linear field map
    f0 = f-df
    f1 = f+df
    #print(f'Generating fieldmap: {f0, f1, shape[1]}')
    x, _ = np.meshgrid(np.linspace(f0, f1, shape[0]), np.linspace(f0, f1, shape[1]))

    # Add noise
    fieldmap: np.ndarray = x 
    fieldmap += np.random.normal(fn_offset * np.random.uniform(-1, 1), fn_sigma, size=shape)
    
    if useRotate:
        rot_angle = rotation * np.random.uniform(-1, 1)
        fieldmap = ndimage.rotate(fieldmap, rot_angle, reshape=False, order=3, mode='nearest')
        
    if useDeform:
        fieldmap = ed.deform_random_grid(fieldmap, sigma=10, points=3, order=3, mode='nearest') # type: ignore
        
    return fieldmap