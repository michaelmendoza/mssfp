import imageio; 
from IPython.display import Video; 

import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import normalized_root_mse

def plot_dataset(data, slice=None, cmap='gray', num_rows=1, figsize=(5, 5), dpi=100):
    ''' 
    Plots a slice of dataset of form: [Slice, Height, Width, Channel]
    
    Parameters:
    -----------
    data : ndarray
        Dataset of form [Slice, Height, Width, Channel]
    slice : int, optional
        Slice index to display (default: 0)
    cmap : str, optional
        Colormap name (default: 'gray')
    num_rows : int, optional
        Number of rows for plot layout (default: 1, single row)
    figsize : tuple, optional
        Figure size (default: (5, 5))
    dpi : int, optional
        Figure dpi (default: 100)
    '''

    slice = 0 if slice is None else slice
    npcs = data.shape[3]
    
    # Calculate number of columns based on number of rows
    num_cols = int(np.ceil(npcs / num_rows))
    
    # Create figure with adjusted layout
    fig, axs = plt.subplots(num_rows, num_cols, figsize=figsize, constrained_layout=True, dpi=dpi)
    
    if cmap:
        plt.set_cmap(cmap)
    
    # Ensure axs is always a 2D array
    if num_rows == 1 and num_cols == 1:
        axs = np.array([[axs]])
    elif num_rows == 1:
        axs = axs.reshape(1, -1)
    elif num_cols == 1:
        axs = axs.reshape(-1, 1)
        
    # Plot data
    for ii in range(npcs):
        row = ii // num_cols
        col = ii % num_cols
        
        _data = np.abs(data[slice, :, :, ii])
        axs[row, col].imshow(_data)
        axs[row, col].set_title(f'{int(ii*(360/npcs))}° PC')
        axs[row, col].set_xticks([])
        axs[row, col].set_yticks([])
    
    # Hide empty subplots if any
    for ii in range(npcs, num_rows * num_cols):
        row = ii // num_cols
        col = ii % num_cols
        axs[row, col].set_visible(False)
    
    # Adjust layout to reduce space between subplots
    plt.tight_layout()
    plt.show()

def combine_channels(data):
    ''' Combines width and channel data i.e. [B, H, W, C] => [B, H, W * C] '''

    _data = np.transpose(data, (3,1,2,0))
    _data = np.transpose(_data, (0,2,1,3))
    _data = np.concatenate(_data, axis=0)
    return np.transpose(_data, (2,1,0))

def show_dataset_channel(data, channel = 4, mp4_filename = '_'):
    ''' Shows a channel of dataset in a video  (time is the slice dimension) '''

    _data = (np.abs(data[:,:,:,channel]) * 255).astype(np.uint8)
    imageio.mimwrite(f'{mp4_filename}.mp4', _data, fps=30); 
    return Video(f'{mp4_filename}.mp4', width=480, height=360) 

def show_dataset(data, mp4_filename = '_'):
    ''' Shows dataset in a video (time is the slice dimension) '''

    if len(data.shape) == 4:
        _data = combine_channels(data)
    elif len(data.shape) == 3:
        _data = data
    else:
        return None

    _data = (np.abs(_data) * 255).astype(np.uint8)
    imageio.mimwrite(f'{mp4_filename}.mp4', _data, fps=30); 
    return Video(f'{mp4_filename}.mp4', width=_data.shape[2], height=_data.shape[1])

def plot_planet_results(mask, T1, T1est, T2, T2est, df, dfest):

    nx, ny = 3, 3
    plt.subplot(nx, ny, 1)
    plt.imshow(T1*mask)
    plt.title('T1 Truth')
    plt.axis('off')

    plt.subplot(nx, ny, 2)
    plt.imshow(T1est)
    plt.title('T1 est')
    plt.axis('off')

    plt.subplot(nx, ny, 3)
    plt.imshow(T1*mask - T1est)
    plt.title('NRMSE: %g' % normalized_root_mse(T1, T1est))
    plt.axis('off')

    plt.subplot(nx, ny, 4)
    plt.imshow(T2*mask)
    plt.title('T2 Truth')
    plt.axis('off')

    plt.subplot(nx, ny, 5)
    plt.imshow(T2est)
    plt.title('T2 est')
    plt.axis('off')

    plt.subplot(nx, ny, 6)
    plt.imshow(T2*mask - T2est)
    plt.title('NRMSE: %g' % normalized_root_mse(T2, T2est))
    plt.axis('off')

    plt.subplot(nx, ny, 7)
    plt.imshow(df*mask)
    plt.title('df Truth')
    plt.axis('off')

    plt.subplot(nx, ny, 8)
    plt.imshow(dfest)
    plt.title('df est')
    plt.axis('off')

    plt.subplot(nx, ny, 9)
    plt.imshow(df*mask - dfest)
    plt.title('NRMSE: %g' % normalized_root_mse(df*mask, dfest))
    plt.axis('off')

    plt.show()

def plot_formatted_data(input, output, results):
    imgs = []
    for n in range(4):
        imgs.append(input[:,:,2*n] + 1j * input[:,:,2*n+1])

    for n in range(4):
        imgs.append(output[:,:,2*n] + 1j * output[:,:,2*n+1])

    for n in range(4):
        imgs.append(results[:,:,2*n] + 1j * results[:,:,2*n+1])

    nx, ny = 3, 4
    plt.figure()
    for ii in range(nx*ny):
        plt.subplot(nx, ny, ii+1)
        plt.imshow(np.abs(imgs[ii]))
    plt.show() 

def plot_real_imag_data(data, index):
    imgs = []
    for n in range(4):
        imgs.append(data[index, :, :, 2*n] + 1j * data[index, :, :, 2*n+1])
    
    nx, ny = 1, 4
    plt.figure()
    for ii in range(nx*ny):
        plt.subplot(nx, ny, ii+1)
        plt.imshow(np.abs(imgs[ii]))
    plt.show() 

def plot_complex_data(data, index):
    imgs = []
    for n in range(data.shape[3]):
        imgs.append(data[index, :, :, n])
    
    nx, ny = 1, data.shape[3]
    plt.figure()
    for ii in range(nx*ny):
        plt.subplot(nx, ny, ii+1)
        plt.imshow(np.abs(imgs[ii]))
    plt.show() 