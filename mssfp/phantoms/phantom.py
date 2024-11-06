from PIL import Image
import math
import numpy as np
import nibabel as nib
import elasticdeform as ed
import random
from scipy import ndimage
from tqdm import tqdm
from glob import glob
import gdown

from ..simulations import ssfp, add_noise_gaussian

def resize(img, size):
    ''' Resizes a mask image. Img should be an uint8. '''
    img = np.squeeze(img).astype(np.uint8)
    new_image = np.array(Image.fromarray(img).resize(size, Image.NEAREST))
    return new_image

def download_brain_data(path: str ='./data'):
    ''' Downloads brain dataset if pathfolder doesn't exists. Data taken from 
    https://brainweb.bic.mni.mcgill.ca/anatomic_normal_20.html '''

    files = glob(f'{path}/*.mnc')
    if len(files) == 0:
        print('Downloading files ...')
        url = 'https://drive.google.com/drive/folders/1oJMmjG44RbpMDkPDNQJzGUlIcdssTYGx?usp=sharing'
        gdown.download_folder(url, quiet=True, output=f'{path}')
        print('Download complete.')

def get_dataset_info(path_data='./data'):
    '''
    Gets information about the available brain atlas data dimensions

    Parameters
    ----------
    path_data : string
        File path for data

    Returns
    -------
    dict
        Dictionary containing:
        - num_images: Number of .mnc files available
        - num_slices: List of number of slices in each image file
        - shapes: List of full shape tuples for each image file
        - total_slices: Total number of slices across all images
    '''
    
    # Download data if needed 
    download_brain_data(path_data)

    # Retrieve file names in dir
    regex = regex='/*.mnc'
    fileList = glob(path_data + regex)
    mnc = [i for i in fileList if 'mnc' in i]
    
    # Get info for each image file
    num_slices = []
    shapes = []
    total_slices = 0
    
    for file in mnc:
        img = nib.load(file)
        data_shape = img.shape
        num_slices.append(data_shape[0])
        shapes.append(data_shape)
        total_slices += data_shape[0]
    
    info = {
        'num_images': len(mnc),
        'num_slices': num_slices,
        'shapes': shapes,
        'total_slices': total_slices
    }
    
    return info

def print_dataset_info(path_data='./data'):
    '''
    Prints a formatted summary of the brain atlas dataset dimensions
    
    Parameters
    ----------
    path_data : string
        File path for data
    '''
    info = get_dataset_info(path_data)
    
    print(f"Dataset Summary:")
    print(f"Number of image files: {info['num_images']}")
    print(f"Total slices across all images: {info['total_slices']}")
    print("\nPer-image details:")
    for i in range(info['num_images']):
        print(f"Image {i}: {info['num_slices'][i]} slices, shape {info['shapes'][i]}")

def generate_ssfp_dataset(N: int = 128, npcs: int = 8, f: float = 1 / 3e-3, 
        TR: float = 3e-3, TE: float = 3e-3 / 2, alpha = np.deg2rad(15), sigma = 0,
        path='./data', data_indices=[], rotate = False, deform = False):
    '''
    SSFP Dataset Generator

    Parameters
    ----------
    N : int
        Grid size.
    npcs: int
        Number of phase cycles.
    f : float
        Off-resonance frequency.
    TR : float
        Repetition time (in seconds). Defaults to 3e-3
    TE : float
        Echo time (in seconds). Defaults to 3e-3 / 2
    alpha : float
        Flip angle in radians.
    sigma : float
        Signal Noise - Generated from gaussian distribution with std dev = sigma 
    path : string
        Folder Path for data defauls to './data'
    '''

    data = load_dataslice(path_data=path, image_index = data_indices[0], slice_index = data_indices[1])
    phantom = generate_3d_phantom(data, N=N, f=f, rotate=rotate, deform=deform)
    M0 = phantom['M0']
    T1 = phantom['t1_map']
    T2 = phantom['t2_map']
    df = phantom['offres']

    # Simulation SSFP with phantom data 
    dataset = []
    pcs = np.linspace(0, 2 * math.pi, npcs, endpoint=False)
    for i in tqdm(range(data.shape[0])):
        M = ssfp(T1[i, :, :], T2[i, :, :], TR, TE, alpha, field_map=df[i, :, :], dphi=pcs, M0=M0[i, :, :])
        M = add_noise_gaussian(M, sigma=sigma)
        dataset.append(M[None, ...])

    dataset = np.concatenate(dataset, axis=0)
    return { 'M': dataset, 'phantom': phantom }

def generate_brain_phantom(N: int = 128, f: float = 1 / 3e-3, path='./data', data_indices=[(None, None), (None, None)], rotate = False, deform = False):
    '''
    SSFP Dataset Generator

    Parameters
    ----------
    N : int
        Grid size.
    f : float
        Off-resonance frequency.
    path : string
        Folder Path for data defauls to './data'
    '''

    data = load_dataslice(path_data=path, image_index = data_indices[0], slice_index = data_indices[1])
    phantom = generate_3d_phantom(data, N=N, f=f, rotate=rotate, deform=deform)
    return phantom

def load_dataslice(path_data='./data', image_index=0, slice_index=150):
    '''
    Loads brain atlas data in mnic1 data format 

    Parameters
    ----------
    path_data : string
        File path for data
    image_index : int or tuple
        Single image index or tuple of (start, end) for range of images
        If tuple includes None, behaves like slice notation:
        (None, end) -> [:end]
        (start, None) -> [start:]
    slice_index : int or tuple 
        Single slice index or tuple of (start, end) for range of slices
        If tuple includes None, behaves like slice notation:
        (None, end) -> [:end]
        (start, None) -> [start:]

    Returns
    -------
    data : ndarray
        Brain atlas data with shape (slices, height, width)
        If single slice, shape is (1, height, width)
    '''
    
    # Download data if needed 
    download_brain_data(path_data)

    # Retrieve file names in dir
    regex = regex='/*.mnc'
    fileList = glob(path_data + regex)
    mnc = [i for i in fileList if 'mnc' in i]

    # Handle single index or tuple for image_index
    if isinstance(image_index, tuple):
        start_img, end_img = image_index
        # Handle None in image_index
        if start_img is None:
            start_img = 0  # Starting from first image (index 0)
        if end_img is None:
            end_img = len(fileList)  # Up to last image
        msg = f"image_index range should be between 0-{len(fileList)-1}"
        assert start_img >= 0 and end_img <= len(fileList), msg
        image_indices = range(start_img, end_img)
    else:
        msg = f"image_index should be between 0-{len(fileList)-1}"
        assert image_index >= 0 and image_index < len(fileList), msg
        image_indices = [image_index]

    # Create slice object from slice_index
    if isinstance(slice_index, tuple):
        start_slice, end_slice = slice_index
        slice_range = slice(start_slice, end_slice)
    else:
        slice_range = slice(slice_index, slice_index + 1)

    # Load and concatenate data for all specified images
    data_list = []
    for idx in image_indices:
        img = nib.load(mnc[idx])
        data = img.get_fdata()[slice_range, :, :].astype(int)
        data[np.where(data >= 4)] = 0  # Only use masks 0-3
        
        # Ensure 3D shape even for single slice
        if len(data.shape) == 2:
            data = data.reshape(1, data.shape[0], data.shape[1])
        data_list.append(data)
    
    # Concatenate along first axis if multiple images
    if len(data_list) > 1:
        data = np.concatenate(data_list, axis=0)
    else:
        data = data_list[0]

    return data

def generate_3d_phantom(data, N: int = 128, f: float = 1 / 3e-3, B0: float = 3, M0: float = 1, rotate=False, deform=False, noise_offset=100, noise_sigma=5):
    ''' 
    Phantom tissue generator

    Parameters
    ----------
        data : Anatomical models generated from .mnc files
        N : Size
        f : Off-resonance
        B0 : Magnetic field
        M0 : Tissue magnetization
    '''

    print('Generating 3d phantom:' + str(data.shape))
    slice_count = data.shape[0]

    # Sample dataset and generate off-resonance 
    sample = np.zeros((slice_count, N, N))
    offres = np.zeros((slice_count, N, N))
    for i in range(slice_count):
        sample[i, :, :] = resize(data[i, :, :], (N, N))
        offres[i, :, :] = generate_offres(N, f=f, rotate=rotate, deform=deform, noise_offset=noise_offset, noise_sigma=noise_sigma) 

    # Generate ROI mask
    roi_mask = (sample != 0)

    # Generate t1/t2 maps
    t1_map, t2_map = create_relaxation_maps(sample, slice_count, N, B0)

    # Package Phantom 
    phantom = {}
    phantom['M0'] = M0 * roi_mask
    phantom['t1_map'] = t1_map * roi_mask
    phantom['t2_map'] = t2_map * roi_mask
    phantom['offres'] = offres * roi_mask
    phantom['mask'] = roi_mask
    phantom['raw'] = sample

    return phantom

def generate_offres(N, f=300, rotate=True, deform=True, max_rot = 360, noise_offset=100, noise_sigma=5):
    '''
    Off-resonance generator

    Parameters
    ----------
    N : int
        Grid size.
    f : float or array_like
        Off-resonance frequency.
    rotate : bool
        Rotation flag
    deform : bool
        Elastic Deformation flag
    max_rot : float
        Maximum rotation angle in degrees
    noise_offset : float
        Noise offset - the range of the mean of the Gaussian noise added to the off-resonance map
    noise_sigma : float
        Noise sigma - the standard deviation of the Gaussian noise added to the off-resonance map
    '''
    
    offres = np.zeros((N, N)) 
    noise_offset = noise_offset * np.random.uniform(-1,1)
    offres_noise = np.random.normal(0, noise_sigma, size=offres.shape)

    rot_angle = max_rot * random.uniform(-1,1)
    offres, _ = np.meshgrid(np.linspace(-f, f, N), np.linspace(-f, f, N))
    offres += offres_noise
    if rotate == True:
        offres = ndimage.rotate(offres, rot_angle, reshape=False, order=3, mode='nearest')
    if deform == True:
        offres = ed.deform_random_grid(offres, sigma=10, points=3, order=3, mode='nearest')
    return offres

def create_relaxation_maps(sample, slice_count, N, B0):
    """
    Creates T1 and T2 relaxation maps for different tissue types.
    
    Args:
        sample: Array containing tissue type identifiers
        slice_count: Number of slices
        N: Dimension size
        B0: Magnetic field strength
        
    Returns:
        t1_map, t2_map: Arrays containing T1 and T2 values for each voxel
    """
    # Tissue parameters mapping: ID -> [T1 formula, T2 value]
    tissue_params = {
        1: ['csf', lambda B: 4.2, lambda B: 1.99],
        2: ['gray-matter', lambda B: 0.857 * (B ** 0.376), lambda B: 0.1],
        3: ['white-matter', lambda B: 0.583 * (B ** 0.382), lambda B: 0.08]
    }
    
    t1_map = np.zeros((slice_count, N, N))
    t2_map = np.zeros((slice_count, N, N))
    
    for tissue_id, (name, t1_func, t2_func) in tissue_params.items():
        mask = (sample == tissue_id)
        t1_map[mask] = t1_func(B0)
        t2_map[mask] = t2_func(B0)
        
    return t1_map, t2_map