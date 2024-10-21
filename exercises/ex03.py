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
    
    # START WORKING HERE
    
def build_gaussian_scalarfield(sz, npts):
    """
    Build the scalar field corresponding to a Gaussian wave function for
    unit cell with edge size "sz" and using a number of sampling points
    "npts".
    """
    r = build_realspace_vectors(sz, npts)
    r -= (5,5,5) # put Gaussian at the center
    r2 = np.einsum('ijkl,ijkl->ijk', r, r)
    
    return (2.0 / np.pi)**(3/4) * np.exp(-r2)
    
def build_realspace_vectors(sz, npts):
    """
    Build the real-space vectors, i.e., the vectors corresponding to the
    sampling points in real-space.
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