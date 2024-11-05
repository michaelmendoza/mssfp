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

    # Load dataslice if data_indices are specifed, otherwise load complete dataset
    if len(data_indices) == 0: 
        data = load_dataset(path)
    else:
        data = load_dataslice(image_index = data_indices[0], slice_index = data_indices[1])

    # Generate phantom
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

def generate_brain_phantom(N: int = 128, f: float = 1 / 3e-3, path='./data', data_indices=[], rotate = False, deform = False):
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

    # Load dataslice if data_indices are specifed, otherwise load complete dataset
    if len(data_indices) == 0: 
        data = load_dataset(path)
    else:
        data = load_dataslice(image_index = data_indices[0], slice_index = data_indices[1])

    # Generate phantom
    phantom = generate_3d_phantom(data, N=N, f=f, rotate=rotate, deform=deform)
    return phantom

def load_dataslice(path_data = './data', image_index = 1, slice_index = 150):
    '''
    Loads brain atlas data in mnic1 data format 

    Parameters
    ----------
    path_data : string
        File path for data
    image_index : int
        Number of images
    slice_index : int
        Slice index
    '''
    
    # Download data if needed 
    download_brain_data(path_data)

    # Retrieve file names in dir
    regex = regex='/*.mnc'
    fileList = glob(path_data + regex)
    mnc = [i for i in fileList if 'mnc' in i]

    # Make image_index is in valid range 
    msg = "image_count should between 1-" + str(len(fileList))
    assert image_index >= 1 and image_index <= len(fileList), msg

    # Load data 
    img = nib.load(mnc[image_index])
    data = img.get_fdata()[slice_index, :, :].astype(int)    
    data[np.where(data >= 4)] = 0 # Only use masks 0-3
    return data.reshape((1, data.shape[0], data.shape[1]))

def load_dataset(path_data = './data', file_count = None, padding = 50):
    '''
    Loads brain atlas data in mnic1 data format and returns an array of
    size (slices, width, height)

    Parameters
    ----------
    path_data : string
        File path for data
    file_count : int
        Number of files to use for dataset (default: None -> All files)
    padding : int
        Number of slices to ignore loading for each file 
    '''

    # Download data if needed 
    download_brain_data(path_data)
    
    # Retrieve file names in dir
    regex = regex='/*.mnc'
    fileList = glob(path_data + regex)
    mnc = [i for i in fileList if 'mnc' in i]

    atlas = []  
    image_count = file_count if file_count else len(fileList)
    for i in range(image_count):
        img = nib.load(mnc[i])
        data = img.get_fdata().astype(int)
        data = data[padding:data.shape[0] - padding] # Remove end slices
        data[np.where(data >= 4)] = 0 # Only use masks 0-3
        atlas.append(data[..., None])
    atlas = np.concatenate(atlas, axis=0)
    atlas = np.squeeze(atlas)

    atlas = atlas.astype(int)
    return atlas

def generate_offres(N, f=300, rotate=True, deform=True):
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
    '''
    max_rot = 360
    offres = np.zeros((N, N))
    rot_angle = max_rot * random.uniform(-1,1)
    offres, _ = np.meshgrid(np.linspace(-f, f, N), np.linspace(-f, f, N))
    if rotate == True:
        offres = ndimage.rotate(offres, rot_angle, reshape=False, order=3, mode='nearest')
    if deform == True:
        offres = ed.deform_random_grid(offres, sigma=10, points=3, order=3, mode='nearest')
    return offres

def generate_3d_phantom(data, N: int = 128, f: float = 1 / 3e-3, B0: float = 3, M0: float = 1, rotate=False, deform=False):
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
        offres[i, :, :] = generate_offres(N, f=f, rotate=rotate, deform=deform) 

    # Generate ROI mask
    roi_mask = (sample != 0)

    # Generate t1/t2 maps
    params = mr_relaxation_parameters(B0)
    t1_map = np.zeros((slice_count, N, N))
    t2_map = np.zeros((slice_count, N, N))
    t1_map[np.where(sample == 1)] = params['csf'][0]
    t1_map[np.where(sample == 2)] = params['gray-matter'][0]
    t1_map[np.where(sample == 3)] = params['white-matter'][0]
    t2_map[np.where(sample == 1)] = params['csf'][1]
    t2_map[np.where(sample == 2)] = params['gray-matter'][1]
    t2_map[np.where(sample == 3)] = params['white-matter'][1]

    # Package Phantom 
    phantom = {}
    phantom['M0'] = M0 * roi_mask
    phantom['t1_map'] = t1_map * roi_mask
    phantom['t2_map'] = t2_map * roi_mask
    phantom['offres'] = offres * roi_mask
    phantom['mask'] = roi_mask
    phantom['raw'] = sample

    return phantom

def generate_phantom(bw_input, alpha, img_no=0, N=128, TR=3e-3, d_flip=10,
            offres=None, B0=3, M0=1):
    ''' 
    phantom generator

    Parameters
    ----------
    bw_input : 
    alpha :
    img_no :
    N : 
    TR :
    d_flip :
    offres :
    B0 :
    M0 : 
    '''

    assert img_no >= 0 and img_no < bw_input.shape[0], "Image index out of bound"

    # these are values from brain web.
    height = bw_input.shape[1]  # X
    width = bw_input.shape[2]  # Y
    dim = 6

    flip_range = np.linspace(alpha - np.deg2rad(d_flip), alpha + np.deg2rad(d_flip), N, endpoint=True)
    flip_map = np.reshape(np.tile(flip_range, N), [N, N]).transpose()
    
    # This is the default off-res map +-300Hz
    if offres is None:
        offres, _ = np.meshgrid(np.linspace(-1 / TR, 1 / TR, N), np.linspace(-1 / TR, 1 / TR, N))
    else:
        offres = offres

    sample = bw_input[img_no, :, :]

    sample = np.reshape(sample, (bw_input.shape[1], bw_input.shape[2]))
    sample = resize(sample, (N, N))
    roi_mask = (sample != 0)
    ph = np.zeros((N, N, dim))

    params = mr_relaxation_parameters(B0)
    t1_map = np.zeros((N, N))
    t2_map = np.zeros((N, N))
    t1_map[np.where(sample == 1)] = params['csf'][0]
    t1_map[np.where(sample == 2)] = params['gray-matter'][0]
    t1_map[np.where(sample == 3)] = params['white-matter'][0]
    t2_map[np.where(sample == 1)] = params['csf'][1]
    t2_map[np.where(sample == 2)] = params['gray-matter'][1]
    t2_map[np.where(sample == 3)] = params['white-matter'][1]

    ph[:, :, 0] = M0 * roi_mask
    ph[:, :, 1] = t1_map * roi_mask
    ph[:, :, 2] = t2_map * roi_mask
    ph[:, :, 3] = flip_map * roi_mask
    ph[:, :, 4] = offres * roi_mask
    ph[:, :, 5] = sample #raw data

    return ph

def get_phantom_parameters(phantom):
    assert phantom.shape[2] == 6, 'Last axes has to be 6!!'

    M0, T1, T2, flip_angle, df, sample = phantom[:, :, 0], \
                                         phantom[:, :, 1], \
                                         phantom[:, :, 2], \
                                         phantom[:, :, 3], \
                                         phantom[:, :, 4], \
                                         phantom[:, :, 5]

    return M0, T1, T2, flip_angle, df, sample


def mr_relaxation_parameters(B0):
    '''Returns MR relaxation parameters for certain tissues.

    Returns
    -------
    params : dict
        Gives entries as [t1, t2]

    Notes
    -----
        Model: T1 = A * B0^C will be used. 
    '''

    t1_t2 = dict()
    t1_t2['csf'] = [4.2, 1.99] #labelled T1 and T2 map for CSF
    t1_t2['gray-matter'] = [.857 * (B0 ** .376), .1] #labelled T1 and T2 map for Gray Matter
    t1_t2['white-matter'] = [.583 * (B0 ** .382), .08] #labelled T1 and T2 map for White Matter
    return t1_t2
