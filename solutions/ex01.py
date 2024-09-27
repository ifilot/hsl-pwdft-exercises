# -*- coding: utf-8 -*-

import numpy as np

def main():
    npts = 64 # number of data points per cartesian direction
    sz = 10   # size of the cubic unit cell
    
    # load datasets
    mo5 = np.load('../data/mo5.npy').reshape(npts, npts, npts)
    mo6 = np.load('../data/mo6.npy').reshape(npts, npts, npts)
    
    # show that the two molecular orbitals are normalized in real-space
    S55 = np.sum(np.power(mo5,2)) * (sz / npts)**3
    S66 = np.sum(np.power(mo6,2)) * (sz / npts)**3
    print('<5|5> = %f' % S55)
    print('<6|6> = %f' % S66)
        
    # show that the two molecular orbitals are orthonormal in real-space
    S56 = np.sum(mo5 * mo6) * (sz / npts)**3
    print('<5|6> = %f' % S56)
    
    # perform FFT transform
    ct = np.sqrt(sz**3) / npts**3
    fft_mo5 = np.fft.fftn(mo5) * ct
    fft_mo6 = np.fft.fftn(mo6) * ct
    
    # calculate reciprocal space integrals
    SF55 = np.einsum('ijk,ijk', fft_mo5, fft_mo5.conjugate())
    SF66 = np.einsum('ijk,ijk', fft_mo5, fft_mo5.conjugate())
    SF56 = np.einsum('ijk,ijk', fft_mo5, fft_mo6.conjugate())
    print('<5|5> = %f' % SF55.real)
    print('<6|6> = %f' % SF66.real)
    print('<5|6> = %f' % SF56.real)
    
if __name__ == '__main__':
    main()