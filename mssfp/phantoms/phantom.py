import math
import numpy as np
from typing import Tuple, Any, List, Union
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
                          padding: int = 8, 
                          slices = 1, 
                          TR = 3e-3, 
                          TE = 3e-3 / 2, 
                          alpha = np.deg2rad(15), 
                          npcs = 4, 
                          sigma = 0.005, 
                          f: float = 1/3e-3,
                          useRotate: bool = False, 
                          useDeform: bool = False, 
                          offres_offset: float = 0, 
                          offres_sigma: float = 0,
                          data_indices: Tuple[int, int] = (0, 150),
                          path: str = './data'):

    if phantom_type == 'brain':
        dataset = brain.BrainDataset(path)
        seg = dataset.load_slice(data_indices[0], data_indices[1])
        generator = brain.PhantomGenerator()
        phantom = generator.generate_3d_phantom(seg, N=shape, f=f, rotate=useRotate, deform=useDeform, 
                                            offres_offset=offres_offset, offres_sigma=offres_sigma)
    else:
        seg = simple.generate_segmentation_masks(shape, ids, padding, phantom_type)
        phantom = simple.generate_phantom(seg, f=f, useRotate=useRotate, useDeform=useDeform, offres_offset=offres_offset, offres_sigma=offres_sigma)

    # Simulation SSFP with phantom data 
    dataset = []
    pcs = np.linspace(0, 2 * math.pi, npcs, endpoint=False)
    for i in tqdm(range(slices)):
        M = ssfp(phantom.t1, phantom.t2, TR, TE, alpha, f0=phantom.f0, field_map=phantom.fieldmap, dphi=pcs, M0=phantom.M0)

        M = add_noise_gaussian(M, sigma=sigma)
        dataset.append(M[None, ...])

    dataset = np.concatenate(dataset, axis=0)
    return { 'M': dataset, **asdict(phantom) }