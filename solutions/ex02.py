# -*- coding: utf-8 -*-

import numpy as np

#
# EXERCISE 2
#

def main():
    npts = 64 # number of data points per cartesian direction
    sz = 10   # size of the cubic unit cell
    
    # load datasets
    mo5 = np.load('../data/mo5.npy')
    mo6 = np.load('../data/mo6.npy')
    
    # perform FFT transform
    ct = np.sqrt(sz**3) / npts**3
    fft_mo5 = np.fft.fftn(mo5) * ct
    fft_mo6 = np.fft.fftn(mo6) * ct 
   
    # determine FFT vectors
    kvec, k2 = build_fft_vectors(10, 64)
    
    # calculate the kinetic energy
    T5 = 0.5 * np.einsum('ijk,ijk,ijk', fft_mo5, k2, fft_mo5.conjugate())
    T6 = 0.5 * np.einsum('ijk,ijk,ijk', fft_mo6, k2, fft_mo6.conjugate())
    print('T5 = %f' % T5.real)
    print('T6 = %f' % T6.real)
    
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