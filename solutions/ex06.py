# -*- coding: utf-8 -*-

#
# EXERCISE 6
#

import numpy as np
import matplotlib.pyplot as plt
import math
import time

def main():
    #
    # define atomic system
    #
    sz = 10   # size of the cubic unit cell   
    atompos = np.array([
        [5.0000, 5.0000, 5.0000],
        [6.1958, 6.1958, 6.1958],
        [3.8042, 3.8042, 6.1958],
        [3.8042, 6.1958, 3.8042],
        [6.1958, 3.8042, 3.8042]
    ])
    
    atomchg = np.array([6,1,1,1,1])
    unitcell = np.identity(3) * sz
    npts = 32
    Omega = sz**3
    deltaV = Omega / npts**3
    
    # grab molecular orbitals
    orbs_fft = np.load('../data/ch4_orbs.npy')
        
    # calculate plane-wave vectors
    k, k2 = build_fft_vectors(sz, npts)
    
    # calculate kinetic energy
    Ekin = np.einsum('lijk,ijk,lijk', orbs_fft, k2, orbs_fft.conjugate()).real
    
    # construct electorn density
    orbs = np.zeros_like(orbs_fft)
    Ct = npts**3 / np.sqrt(sz**3)
    for i in range(len(orbs)):
        orbs[i,:,:,:] = np.fft.ifftn(orbs_fft[i,:,:,:]) * Ct
    rho = 2.0 * np.einsum('lijk,lijk->ijk', orbs, orbs.conjugate()).real
    nelec = np.sum(rho) * deltaV
    
    # check that the number of electrons is sensible
    print('Number of electrons: %f' % nelec)
    
    # construct Hartree potential and calculate electron-electron repulsion
    fft_rho = np.fft.fftn(rho)
    with np.errstate(divide='ignore', invalid='ignore'):
        fft_hartree = 4.0 * np.pi * fft_rho / k2
        fft_hartree[~np.isfinite(fft_hartree)] = 0.0
    hartree = np.fft.ifftn(fft_hartree)
    Erep = 0.5 * np.sum(hartree * rho).real * deltaV
    
    # calculate nuclear attraction
    vpot = calculate_vpot(unitcell, atompos, atomchg, k, k2, Omega)
    Enuc = np.sum(vpot * rho).real * deltaV
    
    # calculate exchange-correlation
    Ex = np.sum(lda_x(rho)[0] * rho) * deltaV
    Ec = np.sum(lda_c_vwn(rho)[0] * rho) * deltaV
    Exc = Ex + Ec
    
    print('Ekin = %6.4f' % Ekin)
    print('Erep = %6.4f' % Erep)
    print('Enuc = %6.4f' % Enuc)
    print('Exc = %6.4f' % Exc)
    
    # calculate Ewald sum
    Eewald = calculate_ewald_sum(unitcell, atompos, atomchg, sz, gcut=2)
    print('Eewald = %6.4f' % Eewald)
    
    Etot = Ekin + Erep + Enuc + Exc + Eewald
    print('Total electronic energy: %6.4f' % Etot)
    
def calculate_ewald_sum(unitcell, atompos, atomchg, sz, gcut=2, gamma=1e-8):
    """
    Calculate the Ewald sum
    """
    # establish alpha value for screening Gaussian charges
    alpha = -0.25 * gcut**2 / np.log(gamma)
    Omega = sz**3

    # subtract spurious self-interaction
    Eself = np.sqrt(alpha / np.pi) * np.sum(atomchg**2)
    
    # subtract the electroneutrality term using a uniform background charge
    Een = np.pi * np.sum(atomchg)**2 / (2 * alpha * Omega)

    # calculate short-range interaction
    Esr = 0
    amag = np.linalg.norm(unitcell, axis=1) # determine unitcell vector magnitudes
    Nmax = np.rint(np.sqrt(-0.5 * np.log(gamma)) / np.sqrt(alpha) / amag + 1.5)
    T = build_indexed_vectors_excluding_zero(Nmax) @ unitcell

    for ia in range(len(atompos)):
        for ja in range(len(atompos)):
            Rij = atompos[ia] - atompos[ja]     # interatomic distance
            ZiZj = atomchg[ia] * atomchg[ja]    # product of charges
            for t in T:   # loop over all unit cell permutations
                R = np.linalg.norm(Rij + t)
                Esr += 0.5 * ZiZj * math.erfc(R * np.sqrt(alpha)) / R
            if ia != ja:  # terms in primary unit cell
                R = np.linalg.norm(Rij)
                Esr += 0.5 * ZiZj * math.erfc(R * np.sqrt(alpha)) / R

    # calculate long-range interaction
    Elr = 0
    B = 2 * np.pi * np.linalg.inv(unitcell.T)    # reciprocal lattice vectors
    bm = np.linalg.norm(B, axis=1)               # vector magnitudes
    s = np.rint(gcut / bm + 1.5)
    G = build_indexed_vectors_excluding_zero(s) @ B    # produce G-vectors
    G2 = np.linalg.norm(G, axis=1)**2   # calculate G-lengths
    pre = 2 * np.pi / Omega * np.exp(-0.25 * G2 / alpha) / G2

    for ia in range(len(atompos)):
        for ja in range(len(atompos)):
            Rij = atompos[ia] - atompos[ja]
            ZiZj = atomchg[ia] * atomchg[ja]
            GR = np.sum(G * Rij, axis=1)
            Elr += ZiZj * np.sum(pre * np.cos(GR)) # discard imaginary values by using cos
    
    Eewald = Elr + Esr - Eself - Een

    return Eewald

def build_indexed_vectors_excluding_zero(s):
    """
    Build a set of incrementing vectors from [-s_i,s_i], exclusing the zero-term
    """
    m1 = np.arange(-s[0], s[0] + 1)
    m2 = np.arange(-s[1], s[1] + 1)
    m3 = np.arange(-s[2], s[2] + 1)
    M = np.transpose(np.meshgrid(m1, m2, m3)).reshape(-1, 3)
    return M[~np.all(M == 0, axis=1)] # remove zero-term

def build_fft_vectors(sz, npts):
    """
    Construct the reciprocal space vectors of the plane waves
    """    
    # calculate plane wave vector coefficients in one dimension
    k = np.fft.fftfreq(npts) * 2.0 * np.pi * (npts / sz)
    
    # construct plane wave vectors
    k3, k2, k1 = np.meshgrid(k, k, k, indexing='ij')
    
    kvec = np.zeros((npts,npts,npts,3))
    kvec[:,:,:,0] = k1
    kvec[:,:,:,1] = k2
    kvec[:,:,:,2] = k3
    
    k2 = np.einsum('ijkl,ijkl->ijk', kvec, kvec)
    
    return kvec, k2

def calculate_vpot(unitcell, atompos, atomchg, k, k2, Omega):
    """Construct the nuclear attraction potential

    Returns:
        np.ndarray: nuclear attraction potential in real-space
    """
    # calculate structure factor
    sf = np.exp(1j * k @ atompos.T)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        nucpotg = -4.0 * np.pi / k2
        nucpotg[0,0,0] = 0

    # produce the nuclear attraction field      
    vpot = np.fft.fftn(np.einsum('ijk,ijkl,l->ijk', 
                                 nucpotg, 
                                 sf, 
                                 atomchg)) / Omega

    return vpot

def lda_x(rho):
    """
    Slater exchange functional, see Parr and Yang pages 154 and 155,
    equations 7.4.5 and 7.4.9
    """
    f = -3 / 4 * (3 / (2 * np.pi))**(2 / 3)
    rs = (3 / (4 * np.pi * rho))**(1 / 3)

    ex = f / rs
    vx = 4 / 3 * ex

    return ex, vx

def lda_c_vwn(rho):
    """
    Vosko-Wilk-Nusair correlation functional, see Parr and Yang page 275
    equation E.27
    """
    A = 0.0621814
    x0 = -0.409286
    b = 13.0720
    c = 42.7198

    rs = (3 / (4 * np.pi * rho))**(1 / 3)

    x = rs**(1/2)
    X = x**2 + b * x + c
    X0 = x0**2 + b * x0 + c
    fx0 = b * x0 / (x0**2 + b * x0 + c)
    tx = 2 * x + b
    Q = (4 * c - b**2)**(1/2)
    atan = np.arctan(Q / (2*x+b))

    ec = A/2 * (np.log(x**2/X) + 2*b/Q * atan - b*x0/X0 * \
                (np.log((x-x0)**2 / X) + 2 * (b + 2*x0) / Q * atan))

    tt = tx**2 + Q**2
    vc = ec - x * A / 12 * (2 / x - tx / X - 4 * b / tt - fx0 * \
        (2 / (x - x0) - tx / X - 4 * (2 * x0 + b) / tt))

    return ec,vc

if __name__ == '__main__':
    main()