import numpy as np
import math
from typing import Tuple, Any, List, Union
from enum import Enum
from . import fieldmap

# T1/T2 values taken from https://mri-q.com/why-is-t1--t2.html
# Tissue Parameter -> tissue_id: (tissue_name, T1, T2, f0)
tissue_parameters = {
    0: ('none', 0, 0, 0),
    1: ('csf', 4.2, 1.99, 0),
    2: ('gray-matter', 0.9, 0.1, 0),
    3: ('white-matter', 0.6, 0.08, 0),
    4: ('muscle', 0.9, .05, 0),
    5: ('liver', 0.5, 0.04, 0), 
    6: ('fat', 0.25, 0.07, 420),   # 420Hz is the average fat signal at 3T - from chemical shift
    7: ('tendon', 0.4, 0.005, 0),
    8: ('proteins', 0.250, 0.001, 0)
}

def get_tissue_parameters(tissue_id: int) -> Tuple[float, float, float]:
    """Get T1 and T2 relaxation times for a given tissue ID."""
    _, t1, t2, f0 = tissue_parameters[tissue_id]
    return t1, t2, f0

class PhantomShape(Enum):
    LINE = 'line'
    BLOCK = 'block'
    CIRCLE = 'circle'

def generate_line_segmentation_mask(length: int = 256, id: int = 1, padding: int = 32):
    s = (1, length) 
    line = np.ones(s) * id

    if padding > 0:
        line[0, -1] = 0
        line[0, -padding:-1] = 0
        line[0, 0:padding] = 0
        
    line.astype(int)
    return line 

def generate_segmentation_masks( shape: int = 256, ids: List[int] = [1, 2, 3, 4], 
                     padding: int = 8, phantom_type: Union[PhantomShape, str] = PhantomShape.BLOCK) -> np.ndarray:
    """Generate a 2D array filled with shapes of specified values.
    
    Args:
        shape (int): Overall shape of the square output array
        ids (list): List of values to use for each shape
        padding (int): Padding between shapes and at edges
        phantom_type (PhantomShape or str): Type of shapes to generate ('block' or 'circle')
    
    Returns:
        numpy.ndarray: 2D array containing the pattern of shapes
    """
    # Convert string to enum if needed
    if isinstance(phantom_type, str):
        phantom_type = PhantomShape(phantom_type)
    
    # Remove 0 from ids if present as 0 will be the background
    ids = [id for id in ids if id != 0]
    
    if phantom_type == PhantomShape.LINE:
        return generate_line_segmentation_mask(length=shape, id=ids[0], padding=padding)

    if not ids:
        return np.zeros((shape, shape))
    
    # Calculate grid dimensions
    num_shapes = len(ids)
    grid_size = math.ceil(math.sqrt(num_shapes))
    
    # Calculate shape size based on grid and padding
    available_space = shape - (grid_size + 1) * padding
    element_size = available_space // grid_size
    
    if element_size <= 0:
        raise ValueError("Shape too small for given number of elements and padding")
    
    # Initialize output array
    output = np.zeros((shape, shape))
    
    # For circles, we need these for the distance calculation
    if phantom_type == PhantomShape.CIRCLE:
        y, x = np.ogrid[:shape, :shape]
        radius = element_size // 2
    
    # Current shape index
    shape_idx = 0
    
    # Calculate starting positions for shapes
    for row in range(grid_size):
        if shape_idx >= len(ids):
            break
            
        for col in range(grid_size):
            if shape_idx >= len(ids):
                break
            
            if phantom_type == PhantomShape.BLOCK:
                # Calculate block position
                start_y = padding + row * (element_size + padding)
                start_x = padding + col * (element_size + padding)
                
                # Place block
                output[start_y:start_y + element_size, 
                      start_x:start_x + element_size] = ids[shape_idx]
            
            else:  # CIRCLE
                # Calculate circle center
                center_y = padding + radius + row * (element_size + padding)
                center_x = padding + radius + col * (element_size + padding)
                
                # Create circle using distance formula
                dist_from_center = np.sqrt((y - center_y)**2 + (x - center_x)**2)
                
                # Place circle
                circle_mask = dist_from_center < radius
                output[circle_mask] = ids[shape_idx]
            
            shape_idx += 1
    
    return output

def generate_phantom(seg: np.ndarray, f: float = 1/3e-3, B0: float = 3, M0: float = 1,
                        useRotate: bool = False, useDeform: bool = False,
                        offres_offset: float = 100, offres_sigma: float = 5):
    """Generate 3D phantom with mask, t1 map, t2 map, f0 mpa, field map, and segmentation mask."""

    _shape = seg.shape
    mask = (seg != 0) * 1

    T1 = np.zeros((_shape[0], _shape[1]))
    T2 = np.zeros((_shape[0], _shape[1]))
    F0 = np.zeros((_shape[0], _shape[1]))

    for tissue_id in tissue_parameters:
            maskIndices = (seg == tissue_id)
            t1, t2, f0 = get_tissue_parameters(tissue_id)
            T1[maskIndices] = t1
            T2[maskIndices] = t2
            F0[maskIndices] = f0
    
    field_map = fieldmap.generate_fieldmap(_shape, f=f, useRotate=useRotate, useDeform=useDeform, rotation=15, noise_offset=offres_offset, noise_sigma=offres_sigma)

    from .phantom import PhantomData
    return PhantomData(M0=M0*mask, t1=T1, t2=T2, f0=F0, fieldmap=field_map, seg=seg, mask=mask)
