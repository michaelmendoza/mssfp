import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import elasticdeform as ed
import random
from scipy import io, ndimage
from tqdm import tqdm

from mri_ssfp import ma_ssfp, add_noise_gaussian

def generate_brain_dataset():
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
    
    return dataset

def data_loader(path_data, image_count=1, slice_index=175):
    '''
    Loads brain atlas data in mnic1 data format 

    Parameters
    ----------
    path_data : string
        File path for data
    image_count : int
        Number of images
    slice_index : int
        Slice index
    '''
    
    # Retrieve file names in dir
    dir_data = path_data
    os.chdir(dir_data)
    fileList = os.listdir()
    msg = "image_count should between 1-" + str(len(fileList))
    assert image_count >= 1 and image_count <= len(fileList), msg

    height = 434
    width = 362

    mnc = [i for i in fileList if 'mnc' in i]
    atlas = np.zeros((image_count, height, width))  
    for i in range(image_count):
        img = nib.load(mnc[i])
        data = img.get_fdata()[slice_index, :, :].astype(int)
        plt.imshow(data)
        plt.show()
        
        data[np.where(data >= 4)] = 0 # Only use masks 0-3
        atlas = data.reshape((1, height, width))

    atlas = atlas.astype(int)
    return atlas

def dataset_loader(path_data):
    '''
    Loads brain atlas data in mnic1 data format and returns an array of
    size (slices, width, height)

    Parameters
    ----------
    path_data : string
        File path for data
    '''
    
    # Retrieve file names in dir
    dir_data = path_data
    os.chdir(dir_data)
    fileList = os.listdir()
    image_count = len(fileList)

    height = 434
    width = 362

    mnc = [i for i in fileList if 'mnc' in i]
    atlas = []  
    for i in range(image_count):
        img = nib.load(mnc[i])
        data = img.get_fdata().astype(int)
        data[np.where(data >= 4)] = 0 # Only use masks 0-3
        atlas.append(data[..., None])
    atlas = np.concatenate(atlas, axis=0)
    atlas = np.squeeze(atlas)

    atlas = atlas.astype(int)
    print(atlas.shape)
    return atlas

def generate_offres(N, f=300, rotate=True, deform=True):
    '''
    offres generator

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
    sample = cv2.resize(sample, (N, N), interpolation=cv2.INTER_NEAREST)
    roi_mask = (sample != 0)
    ph = np.zeros((N, N, dim))

    params = _mr_relaxation_parameters(B0)
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

def _mr_relaxation_parameters(B0):
    '''Returns MR relaxation parameters for certain tissues.

    Returns
    -------
    params : dict
        Gives entries as [A, C, (t1), t2, chi]

    Notes
    -----
    If t1 is None, the model T1 = A*B0^C will be used.  If t1 is not
    np.nan, then specified t1 will be used.
    '''

    # params['tissue-name'] = [A, C, (t1 value if explicit), t2, chi]
    # params = dict()
    # params['scalp'] = [.324, .137, np.nan, .07, -7.5e-6]
    # params['marrow'] = [.533, .088, np.nan, .05, -8.85e-6]
    # params['csf'] = [np.nan, np.nan, 4.2, 1.99, -9e-6]
    # params['blood-clot'] = [1.35, .34, np.nan, .2, -9e-6]
    # params['gray-matter'] = [.857, .376, np.nan, .1, -9e-6]
    # params['white-matter'] = [.583, .382, np.nan, .08, -9e-6]
    # params['tumor'] = [.926, .217, np.nan, .1, -9e-6]

    t1_t2 = dict()
    t1_t2['csf'] = [4.2, 1.99] #labelled T1 and T2 map for CSF
    t1_t2['gray-matter'] = [.857 * (B0 ** .376), .1] #labelled T1 and T2 map for Gray Matter
    t1_t2['white-matter'] = [.583 * (B0 ** .382), .08] #labelled T1 and T2 map for White Matter
    return t1_t2