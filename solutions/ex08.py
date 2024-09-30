# -*- coding: utf-8 -*-

#
# EXERCISE 8
#

from pypwdft import SystemBuilder, PyPWDFT
import numpy as np
import matplotlib.pyplot as plt

def main():
    sz = 10
    npts = 32
    s = SystemBuilder().from_name('nh3', sz, npts)
    res = PyPWDFT(s).scf(verbose=True)
    
    fig, ax = plt.subplots(3, 4, dpi=144, figsize=(12,8))
    for i in range(0,4):
        limit = np.max(np.abs(res['orbc_rs'][i,npts//2,:,:].real))
        ax[0,i].imshow(res['orbc_rs'][i,npts//2,:,:].real, extent=(0,sz,0,sz), 
                       interpolation='bicubic', cmap='PiYG',
                       vmin=-limit, vmax=limit)
        limit = np.max(np.abs(res['orbc_rs'][i,npts//2,:,:].imag))
        ax[1,i].imshow(res['orbc_rs'][i,npts//2,:,:].imag, extent=(0,sz,0,sz), 
                       interpolation='bicubic', cmap='PiYG',
                       vmin=-limit, vmax=limit)
        ax[2,i].imshow(np.power(res['orbc_rs'][i,npts//2,:,:],2).real, 
                       extent=(0,sz,0,sz), interpolation='bicubic')
    
        for j in range(0,3):
            ax[j,i].set_xlabel(r'$x$ [a.u.]')
            ax[j,i].set_ylabel(r'$y$ [a.u.]')
            
        ax[0,i].set_title(r'$\mathbb{R}\;[\psi_{%i}]$' % (i+1))
        ax[1,i].set_title(r'$\mathbb{I}\;[\psi_{%i}]$' % (i+1))
        ax[2,i].set_title(r'$\rho_{%i}$' % (i+1))

    plt.tight_layout()

if __name__ == '__main__':
    main()