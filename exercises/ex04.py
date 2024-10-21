# -*- coding: utf-8 -*-

#
# EXERCISE 4
#

import numpy as np
import matplotlib.pyplot as plt

def main():
    npts = 64 # number of data points per cartesian direction
    sz = 10   # size of the cubic unit cell
    
    # create scalar field
    psi = build_gaussian_scalarfield(sz, npts)
    rho = np.power(psi,2)
    
    # START WORKING HERE
    
def build_gaussian_scalarfield(sz, npts):
    """
    Build the (normalized) wave function corresponding to a Gaussian
    """
    r = build_realspace_vectors(sz, npts)
    r -= (5,5,5) # put Gaussian at the center
    r2 = np.einsum('ijkl,ijkl->ijk', r, r)
    
    return (2.0 / np.pi)**(3/4) * np.exp(-r2)
    
def build_hartree_potential(rho, sz, npts):
    """
    Build the Hartree potential from the electron density
    """
    # build fft vectors
    kvec, k2 = build_fft_vectors(sz, npts)
    
    fft_rho = np.fft.fftn(rho)
    with np.errstate(divide='ignore', invalid='ignore'):
        fft_hartree = 4.0 * np.pi * fft_rho / k2
        fft_hartree[~np.isfinite(fft_hartree)] = 0.0
    
    return np.fft.ifftn(fft_hartree)

def build_external_potential(npts, sz):
    """
    Calculate the external potential by single nucleus with charge Z=1
    """
    # build fft vectors
    kvec, k2 = build_fft_vectors(sz, npts)
    R = (sz/2, sz/2, sz/2)
    
    # generate structure factor and nuclear attraction field
    sf = np.exp(-1j * kvec @ R) / np.sqrt(sz**3)
    ct = np.sqrt(sz**3) / npts**3
    with np.errstate(divide='ignore', invalid='ignore'):
        nupotg = -4.0 * np.pi / k2
        nupotg[0,0,0] = 0

    vnuc = np.fft.ifftn(sf * nupotg) / ct
    
    return vnuc

def build_realspace_vectors(sz, npts):
    """
    Construct the set of sampling points in real space
    """
    # determine grid points in real space
    c = np.linspace(0, sz, npts, endpoint=False)

    # construct real space sampling vectors
    z, y, x = np.meshgrid(c, c, c, indexing='ij')   
    
    r = np.zeros((npts,npts,npts,3))
    r[:,:,:,0] = x
    r[:,:,:,1] = y
    r[:,:,:,2] = z
    
    return r    

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
    
if __name__ == '__main__':
    main()