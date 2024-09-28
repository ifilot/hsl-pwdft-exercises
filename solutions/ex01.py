# -*- coding: utf-8 -*-

#
# EXERCISE 1
#

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

def main():
    npts = 64 # number of data points per cartesian direction
    sz = 10   # size of the cubic unit cell
    
    # load datasets
    mo5 = np.load('../data/mo5.npy').reshape(npts, npts, npts)
    mo6 = np.load('../data/mo6.npy').reshape(npts, npts, npts)
    
    # visualize the two molecular orbitals
    fig, ax = plt.subplots(1, 2, dpi=144)
    im1 = ax[0].imshow(mo5[:, npts//2, :], origin='lower', extent=(0,10,0,10))
    im2 = ax[1].imshow(mo6[:, :, npts//2], origin='lower', extent=(0,10,0,10))
    
    divider = make_axes_locatable(ax[0])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im1, cax=cax, orientation='vertical')
    
    divider = make_axes_locatable(ax[1])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im2, cax=cax, orientation='vertical')
    
    plt.tight_layout()
    
    ###########################################################################
    
    # show that the two molecular orbitals are normalized in real-space
    S55 = np.sum(np.power(mo5,2)) * (sz / npts)**3
    S66 = np.sum(np.power(mo6,2)) * (sz / npts)**3
    print('<5|5> = %f' % S55)
    print('<6|6> = %f' % S66)
        
    # show that the two molecular orbitals are orthonormal in real-space
    S56 = np.sum(mo5 * mo6) * (sz / npts)**3
    print('<5|6> = %f' % S56)
    
    ###########################################################################
    
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