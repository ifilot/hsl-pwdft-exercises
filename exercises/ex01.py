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
    
if __name__ == '__main__':
    main()