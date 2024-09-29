# -*- coding: utf-8 -*-

#
# EXERCISE 7
#

import numpy as np
import scipy as sc

def main():
    H = np.load('../data/co_fock.npy')
    
    # calculate eigenvalues and eigenvectors using conventional (full) matrix diagonalization
    e1,v1 = np.linalg.eigh(H)
    
    # calculate first 7 eigenvalues and vectors using Arnoldi method
    e2,v2 = sc.sparse.linalg.eigsh(H, k=7, which='SA')

    # print the differences between full matrix diagonalization and the Arnoldi
    # method for the first 7 vectors
    for i in range(len(e2)):
        print('%i: %+12.10f' % (i+1, e2[i] - e1[i]))

if __name__ == '__main__':
    main()