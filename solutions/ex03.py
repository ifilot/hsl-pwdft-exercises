# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

#
# EXERCISE 3
#

def main():
    npts = 64 # number of data points per cartesian direction
    sz = 10   # size of the cubic unit cell
    
    # create scalar field
    psi = build_gaussian_scalarfield(sz, npts)
    
    # test real-space normalization
    S = (sz / npts)** 3 * np.sum(np.power(psi,2))
    print('<psi|psi> = %f' % S)
    
    ###########################################################################
    
    # construct the electron density in reciprocal space
    rho = np.power(psi,2)
    fft_rho = np.fft.fftn(rho)
    
    ###########################################################################
    
    # build fft vectors
    kvec, k2 = build_fft_vectors(sz, npts)
    
    # build the Hartree potential in reciprocal space
    with np.errstate(divide='ignore', invalid='ignore'):
        fft_hartree = 4.0 * np.pi * fft_rho / k2
        fft_hartree[~np.isfinite(fft_hartree)] = 0.0
    
    ###########################################################################
    
    # perform inverse FFT
    hartree = np.fft.ifftn(fft_hartree)
    
    ###########################################################################
    
    # calculate electron-electron repulsion energy in real-space
    Eee = np.einsum('ijk,ijk', hartree, rho) * (sz / npts)** 3
    print('Eee = %f (%f)' % (Eee.real, 2 / np.sqrt(np.pi)))
    
def build_gaussian_scalarfield(sz, npts):
    r = build_realspace_vectors(sz, npts)
    r -= (5,5,5) # put Gaussian at the center
    r2 = np.einsum('ijkl,ijkl->ijk', r, r)
    
    return (2.0 / np.pi)**(3/4) * np.exp(-r2)
    
def build_realspace_vectors(sz, npts):
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