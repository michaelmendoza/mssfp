''' ssfp simulation '''

from typing import Any, Union
import numpy as np

def ssfp(T1: Union[float, np.ndarray],
         T2: Union[float, np.ndarray],
         TR: float,
         TE: float,
         alpha: float,
         dphi: Union[float, np.ndarray] = (0,),
         field_map: Union[float, np.ndarray] = 0,
         M0: Union[float, np.ndarray] = 1,
         f0: Union[float, np.ndarray] = 0,
         phi: Union[float, np.ndarray] = 0,
         useSqueeze: bool = True) -> np.ndarray:
    """ Transverse signal for SSFP MRI after excitation at TE.

    Parameters
    ----------
    T1 : float or np.ndarray
        longitudinal exponential decay time constant (in seconds).
    T2 : float or np.ndarray
        transverse exponential decay time constant (in seconds).
    TR : float
        repetition time (in seconds).
    TE : float
        echo time (in seconds).
    alpha : float 
        flip angle (in rad).
    dphi : float or np.ndarray, optional
        Linear phase-cycle increment (in rad).
    field_map : float or np.ndarray, optional
        B0 field map (in Hz).
    M0 : float or np.ndarray, optional
        proton density.
    f0 : float or np.ndarray, optional
        off-resonance (in Hz). Includes factors like the chemical shift
        of species w.r.t. the water peak.
    phi : float or np.ndarray, optional
        phase offset (in rad).
    useSqueeze : bool, optional
        Whether to squeeze the output array.

    Returns
    -------
    np.ndarray
        Complex-valued array representing the transverse magnetization.
    """

    # Convert inputs to arrays and adjust field map convention
    inputs = [T1, T2, f0, -field_map, M0]
    T1, T2,  f0, field_map, M0 = map(np.asarray, inputs)
    dphi = np.asarray(dphi).ravel()

    # Determine the broadcasted shape and reshape dphi
    broadcast_shape = np.broadcast(*inputs).shape
    dphi = dphi.reshape((1,) * len(broadcast_shape) + (-1,))

    # Broadcast all input arrays to the common shape
    T1, T2, f0, field_map, M0 = np.broadcast_arrays(*inputs)

    # Compute exponential decays
    E1 = np.where(T1 > 0, np.exp(-TR / T1), 0)
    E2 = np.where(T2 > 0, np.exp(-TR / T2), 0)

    # Compute beta and trigonometric functions
    beta = 2 * np.pi * (f0 + field_map) * TR
    cos_alpha, sin_alpha = np.cos(alpha), np.sin(alpha)

    # Prepare for broadcasting with dphi
    expand_dims = lambda x: x[..., np.newaxis]
    beta, E1, E2, sin_alpha, cos_alpha, M0, T2 = map(expand_dims, 
                                                          [beta, E1, E2, sin_alpha, cos_alpha, M0, T2])

    # Compute theta
    theta = beta - dphi
    cos_theta, sin_theta = np.cos(theta), np.sin(theta)

    # Calculate denominator
    denominator = ((1 - E1 * cos_alpha) * (1 - E2 * cos_theta) -
                   E2 * (E1 - cos_alpha) * (E2 - cos_theta))

    # Calculate Mx and My
    common_factor = M0 * (1 - E1) * sin_alpha / denominator
    Mx = common_factor * (1 - E2 * cos_theta)
    My = common_factor * E2 * sin_theta

    # Combine Mx and My into complex magnetization and apply phase and T2 decay
    _phi = beta * (TE / TR) + phi
    T2_decay = np.where(T2 > 0, np.exp(-TE / T2), 0)
    M = ( Mx + 1j * My) * np.exp(1j * _phi) * T2_decay

    return np.squeeze(M) if useSqueeze else M


def add_noise_gaussian(I, mu=0, sigma=0.005, factor = 1):
    '''add gaussian noise to given simulated bSSFP signals
    Parameters
    ----------
    I: array_like
       images size(M,N,C)
    mu: float
        mean of the normal distribution
    sd: float
        standard deviation of the normal distribution

    Returns
    -------
    Mxy : numpy.array
        Transverse complex magnetization with added .
    '''
    noise = factor * np.random.normal(mu, sigma, (2,) + np.shape(I))
    noise_matrix = noise[0]+ 1j*noise[1]
    return I + noise_matrix
    