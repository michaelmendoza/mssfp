import gdown
import nibabel as nib
import numpy as np
from PIL import Image
from glob import glob
from typing import Dict, Optional, Tuple, Union
from . import fieldmap

class TissueParameters:
    """Tissue relaxation parameters for different tissue types."""
    
    def __init__(self, B0: float):
        self.parameters = {
            1: ('csf', lambda _: 4.2, lambda _: 1.99),
            2: ('gray-matter', lambda B: 0.857 * (B ** 0.376), lambda _: 0.1),
            3: ('white-matter', lambda B: 0.583 * (B ** 0.382), lambda _: 0.08)
        }
        self.B0 = B0

    def get_relaxation_times(self, tissue_id: int) -> Tuple[float, float]:
        """Get T1 and T2 relaxation times for a given tissue ID."""
        _, t1_func, t2_func = self.parameters[tissue_id]
        return t1_func(self.B0), t2_func(self.B0)

class BrainDataset:
    """Handles brain dataset operations including downloading and loading."""
    
    def __init__(self, path: str = './data'):
        self.path = path
        self._ensure_data_exists()
        
    def _ensure_data_exists(self) -> None:
        """Download brain dataset if it doesn't exist. Data taken from https://brainweb.bic.mni.mcgill.ca/anatomic_normal_20.html"""
        if not glob(f'{self.path}/*.mnc'):
            print('Downloading brain dataset...')
            url = 'https://drive.google.com/drive/folders/1oJMmjG44RbpMDkPDNQJzGUlIcdssTYGx?usp=sharing'
            gdown.download_folder(url, quiet=True, output=self.path)
            print('Download complete.')

    def __repr__(self) -> str:
        info = self.get_info()
        return f'BrainDataset(path={self.path}, num_images={info["num_images"]}, total_slices={info["total_slices"]})'

    def print_info(self) -> str:
        info = self.get_info()

        print(f"Dataset Summary:")
        print(f"Number of image files: {info['num_images']}")
        print(f"Total slices across all images: {info['total_slices']}")
        print("\nPer-image details:")
        for i in range(info['num_images']):
            print(f"Image {i}: {info['num_slices'][i]} slices, shape {info['shapes'][i]}")

    def get_info(self) -> Dict:
        """Get dataset information including dimensions and slice counts."""
        mnc_files = [f for f in glob(f'{self.path}/*.mnc')]
        
        shapes = []
        num_slices = []
        total_slices = 0
        
        for file in mnc_files:
            img = nib.load(file)
            shape = img.shape
            shapes.append(shape)
            num_slices.append(shape[0])
            total_slices += shape[0]
            
        return {
            'num_images': len(mnc_files),
            'num_slices': num_slices,
            'shapes': shapes,
            'total_slices': total_slices
        }

    def load_slice(self, 
                  image_index: Union[int, Tuple[Optional[int], Optional[int]]] = 0,
                  slice_index: Union[int, Tuple[Optional[int], Optional[int]]] = 150) -> np.ndarray:
        """Load brain atlas data slices."""
        mnc_files = sorted(glob(f'{self.path}/*.mnc'))
        
        # Handle image index range
        if isinstance(image_index, tuple):
            start_img, end_img = image_index
            start_img = start_img if start_img is not None else 0
            end_img = end_img if end_img is not None else len(mnc_files)
            image_indices = range(start_img, end_img)
        else:
            image_indices = [image_index]
            
        # Create slice range
        if isinstance(slice_index, tuple):
            slice_range = slice(slice_index[0], slice_index[1])
        else:
            slice_range = slice(slice_index, slice_index + 1)
            
        data_list = []
        for idx in image_indices:
            img = nib.load(mnc_files[idx])
            data = img.get_fdata()[slice_range, :, :].astype(int)
            data[data >= 4] = 0  # Only use masks 0-3
            
            if len(data.shape) == 2:
                data = data.reshape(1, *data.shape)
            data_list.append(data)
            
        return np.concatenate(data_list, axis=0) if len(data_list) > 1 else data_list[0]

class PhantomGenerator:
    """Generates SSFP phantoms with specified parameters."""
    
    @staticmethod
    def resize_mask(img: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
        """Resize a mask image using nearest neighbor interpolation."""
        return np.array(Image.fromarray(np.squeeze(img).astype(np.uint8)).resize(size, Image.NEAREST))

    def generate_3d_phantom(self, data: np.ndarray, N: int = 128, 
                          f: float = 0, df: float = 1/3e-3, df_window: float = 0.0,
                          fn_offset: float = 100, fn_sigma: float = 5,
                          add_perlin: bool = False, perlin_size: int = 2, perlin_scale: float = 0.1,
                          rotation: float = 360, useRotate: bool = False, useDeform: bool = False,
                           B0: float = 3, M0: float = 1):
        """Generate 3D phantom with tissue properties."""
        slice_count = data.shape[0]

        # Initialize arrays
        sample = np.zeros((slice_count, N, N))
        offres = np.zeros((slice_count, N, N))
        
        # Process each slice
        for i in range(slice_count):
            sample[i] = self.resize_mask(data[i], (N, N))
            _f = _f if df_window == 0 else f * np.random.uniform(1 - df_window, 1 + df_window)
            _df = df if df_window == 0 else df * np.random.uniform(1 - df_window, 1 + df_window)
            offres[i] = fieldmap.generate_fieldmap((N, N), _f, _df, fn_offset=fn_offset, fn_sigma=fn_sigma, 
                                                   add_perlin=add_perlin, perlin_size=perlin_size, perlin_scale=perlin_scale,
                                                   rotation=rotation, useRotate=useRotate, useDeform=useDeform)

        # Generate masks and maps
        roi_mask = (sample != 0)
        tissue_params = TissueParameters(B0)
        t1_map = np.zeros((slice_count, N, N))
        t2_map = np.zeros((slice_count, N, N))
        
        for tissue_id in tissue_params.parameters:
            mask = (sample == tissue_id)
            t1, t2 = tissue_params.get_relaxation_times(tissue_id)
            t1_map[mask] = t1
            t2_map[mask] = t2

        from .phantom import PhantomData
        return PhantomData(
            M0=M0 * roi_mask,
            t1=t1_map * roi_mask,
            t2=t2_map * roi_mask,
            f0=0 * roi_mask,
            fieldmap=offres * roi_mask,
            mask=roi_mask,
            seg=sample
        )
