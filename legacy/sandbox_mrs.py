import numpy as np
import matplotlib.pyplot as plt 
from mri_ssfp import ssfp, ma_ssfp

def flip_angles_SSFP():
    N = 1024
    T1, T2 = 1, .5
    TR = 3e-3
    TE = TR / 2.0
    beta = np.linspace(-np.pi, np.pi, N)
    f = beta / TR / (2 * np.pi)

    M = np.zeros( (N, 4), dtype=complex)
    M[:, 0] = ssfp(T1, T2, TR, TE, np.deg2rad(0.5), f0=f)
    M[:, 1] = ssfp(T1, T2, TR, TE, np.deg2rad(1), f0=f)
    M[:, 2] = ssfp(T1, T2, TR, TE, np.deg2rad(10), f0=f)
    M[:, 3] = ssfp(T1, T2, TR, TE, np.deg2rad(30), f0=f)

    print(M.shape)

    plt.subplot(211)
    plt.plot(f, np.absolute(M))
    plt.ylabel('Magitude')
    plt.title('SSPF Sequence')
    plt.gca().legend(('0.5','1','10','30'))
    plt.grid(True)

    plt.subplot(212)
    plt.plot(f, np.angle(M))
    plt.xlabel('Off-Resonance (Hz)')
    plt.ylabel('Phase')
    plt.gca().legend(('0.5','1','10','30'))
    plt.grid(True)
    plt.show()

def low_flip_angle_SSFP():
    N = 1024
    npcs = 16
    T1, T2 = 1, .5
    TR, alpha = 3e-3, np.deg2rad(5)
    TE = TR / 2.0
    pcs = np.linspace(0, 2*np.pi, npcs, endpoint=False)
    BetaMax = np.pi
    beta = np.linspace(-BetaMax, BetaMax, N)
    f = beta / TR / (2 * np.pi)
    M = ma_ssfp(T1, T2, TR, TE, alpha, f0=f, dphi=pcs)
    M = np.reshape(M, (N, npcs))
    print(M.shape)

    mag = np.absolute(M)
    phase = np.angle(M)

    plt.subplot(211)
    plt.plot(f, mag)
    plt.ylabel('Magitude')
    plt.title('SSPF Sequence')
    plt.grid(True)

    plt.subplot(212)
    plt.plot(f, phase)
    plt.xlabel('Off-Resonance (Hz)')
    plt.ylabel('Phase')
    plt.grid(True)
    plt.show()

flip_angles_SSFP()
low_flip_angle_SSFP()