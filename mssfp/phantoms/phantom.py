import math
import numpy as np
from typing import Tuple, Any, List, Union, Optional
from tqdm import tqdm
from dataclasses import dataclass, asdict
from . import simple, brain
from ..simulations import ssfp, add_noise_gaussian

@dataclass
class PhantomData:
    """Container for phantom data with tissue properties and maps."""
    M0: np.ndarray
    t1: np.ndarray
    t2: np.ndarray
    f0: np.ndarray
    fieldmap: np.ndarray
    mask: np.ndarray
    seg: np.ndarray

def generate_ssfp_dataset(phantom_type: str = 'block', 
                          shape: int = 256, 
                          ids: List[int] = [1, 2, 3, 4], 
                          tissues: Optional[Union[dict, None]] = None,
                          padding: int = 8, 
                          slices = 1, 
                          TR = 3e-3, 
                          TE = 3e-3 / 2, 
                          alpha = np.deg2rad(15), 
                          npcs = 4, 
                          sigma = 0.005, 
                          f: float = 0,
                          df: float = 1/3e-3,
                          fn_offset: float = 0, 
                          fn_sigma: float = 0,
                          rotation: float = 0,
                          useRotate: bool = False, 
                          useDeform: bool = False, 
                          data_indices: Tuple[Any, Any] = ((None, None), (None,None)),
                          path: str = './data'):

    # Generate phantom
    print('Generating phantom...')

    # Save function args as settings dict for dataset (use function args as keys)
    args = locals()
    
    if phantom_type == 'brain':
        slices = 1 # Using 3D datasets instead of 2d slices 
        dataset = brain.BrainDataset(path)
        seg = dataset.load_slice(data_indices[0], data_indices[1])
        generator = brain.PhantomGenerator()
        phantom = generator.generate_3d_phantom(seg, N=shape, f=f, df=df, fn_offset=fn_offset, fn_sigma=fn_sigma, 
                                                rotation=rotation, useRotate=useRotate, useDeform=useDeform)
    else:
        seg = simple.generate_segmentation_masks(shape, ids, padding, phantom_type)
        phantom = simple.generate_phantom(seg, slices=slices, f=f, df=df, fn_offset=fn_offset, fn_sigma=fn_sigma, 
                                          rotation=rotation, useRotate=useRotate, useDeform=useDeform, tissues=tissues)

    print('Generating SSFP dataset...')

    # Simulation SSFP with phantom data 
    dataset = []
    pcs = np.linspace(0, 2 * math.pi, npcs, endpoint=False)
    M = ssfp(phantom.t1, phantom.t2, TR, TE, alpha, f0=phantom.f0, field_map=phantom.fieldmap, dphi=pcs, M0=phantom.M0, useSqueeze=False)
    if sigma > 0:
        M = add_noise_gaussian(M, sigma=sigma)

    print('Dataset complete.')
    print(f'Dataset shape: {M.shape}')

    return { 'M': M, 'settings': args, **asdict(phantom) }