'''Estimate off-resonance using two TR ellipses.'''

import numpy as np
from ellipsinator import rotate_points
from tqdm import trange
from ssfp import gs_recon


def _ormtre6(I0: np.ndarray, I1: np.ndarray, mask: np.ndarray, TR0: float, TR1: float, rad: bool) -> np.ndarray:
    if I0.ndim == 1:  # handle single pixel case inelagantly
        I0 = I0[None, :]
        I1 = I1[None, :]
    sh = I0.shape[:-1]
    I0 = np.reshape(I0[mask], (-1, I0.shape[-1]))
    I1 = np.reshape(I1[mask], (-1, I1.shape[-1]))
    ctr0 = gs_recon(I0, pc_axis=-1, second_pass=False)

    # find where circle intersects line, this is ctr1 (or there-abouts)
    # use equation of centered circle and line:
    #     x^2 + y^2 = r^2
    #     r = |GS(I0)|, where GS(I) is the geometric center of ellipse I
    #     (x0, y0) = 0 deg phase cycle of second TR ellipse
    #     (x1, y1) = 180 deg phase cycle of second TR ellipse
    #     y = mx + b
    #     m = (y1 - y0)/(x1 - x0)
    #     b = y - mx = y0 - m*x0 = y1 - m*x1
    #     x^2 + (mx + b) = r^2
    #     => x =  (sqrt((m^2 + 1)r^2 - b^2) - bm)/(m^2 + 1)
    #         OR -(sqrt((m^2 + 1)r^2 - b^2) + bm)/(m^2 + 1)
    #        Choose the smaller rotation, i.e. min |x|
    #     y = mx + b
    #     (x, y) is now GS(I1)

    r2 = np.abs(ctr0)**2
    m = (I1[:, 0].imag - I1[:, 1].imag)/(I1[:, 0].real - I1[:, 1].real)
    m2 = m**2
    b = (I1[:, 0].imag - m*I1[:, 0].real + I1[:, 1].imag - m*I1[:, 1].real)/2
    x = (np.sqrt((m2 + 1)*r2 - b**2) - b*m)/(m2 + 1)
    xalt = -1*(np.sqrt((m2 + 1)*r2 - b**2) + b*m)/(m2 + 1)
    idx = np.abs(xalt) < np.abs(x)
    x[idx] = xalt[idx]
    y = m*x + b
    theta = np.zeros(sh)
    theta[mask] = np.angle(ctr0*np.conj(x + 1j*y))
    if rad:
        # return radians instead of hz
        return theta
    return 1/(TR1/TR0 - 1)*theta/(np.pi*TR0)


def _ormtre8(I0: np.ndarray, I1: np.ndarray, TR0: float, TR1: float, rad: bool) -> np.ndarray:
    # Find geometric centers of both ellipses
    ctr0 = gs_recon(I0, pc_axis=-1, second_pass=False)
    ctr1 = gs_recon(I1, pc_axis=-1, second_pass=False)
    theta = np.angle(ctr0*np.conj(ctr1))
    mult = TR1/TR0
    if mult < 1:  # this should never happen...
        fac = -1/2
    else:
        fac = 1
    if rad:
        # return radians instead of hz
        return theta
    return np.array(1/(mult - 1)*theta/(fac*np.pi*TR0))


def ormtre(I0: np.ndarray, I1: np.ndarray, mask: np.ndarray,
           TR0: float, TR1: float, pc_axis=-1, rad=False) -> float:
    '''Off-resonance using multiple TR ellipses.

    Parameters
    ----------
    I0 : array_like
        Complex-valued phase-cycled pixels with phase-cycles
        corresponding to [0, 90, 180, 270] degrees and TR0.
    I1 : array_like
        Complex-valued phase-cycled pixels with phase-cycles
        corresponding to [0, 180] or [0, 90, 180, 270]
        degrees and TR1.
    TR0, TR1 : float
        TR values in seconds corresponding to I0 and I1.
    pc_axis : int, optional
        Axis holding phase-cycle data.

    Returns
    -------
    theta : array_like
        Off-resonance estimate (Hz).

    Notes
    -----
    Uses 6 or 8 phase-cycled images to estimate off-resonance.
    '''

    I0 = np.moveaxis(I0, pc_axis, -1)
    I1 = np.moveaxis(I1, pc_axis, -1)
    assert I0.shape[-1] == 4, 'I0 must have 4 phase-cycles!'
    assert I1.shape[-1] in {2, 4}, 'I1 must have 2 or 4 phase-cycles!'

    if I1.shape[-1] == 2:
        return _ormtre6(I0, I1, mask, TR0, TR1, rad)
    return _ormtre8(I0, I1, TR0, TR1, rad) # TODO: add mask


if __name__ == '__main__':
    pass