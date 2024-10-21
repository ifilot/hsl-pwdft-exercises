# -*- coding: utf-8 -*-

#
# EXERCISE 5
#

import numpy as np
import matplotlib.pyplot as plt
import math
import time

def main():
    sz = 10   # size of the cubic unit cell
    
    unitcell = np.identity(3) * sz
    atompos = np.array([
        [5.0000,  5.0000,  5.9677],
        [5.0000,  5.0000,  3.7097]
    ])
    atomchg = np.array([8,6])
    
    # START WORKING HERE
    
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

    return Eewald, Nmax, s

def build_indexed_vectors_excluding_zero(s):
    """
    Build a set of incrementing vectors from [-s_i,s_i], exclusing the zero-term
    """
    m1 = np.arange(-s[0], s[0] + 1)
    m2 = np.arange(-s[1], s[1] + 1)
    m3 = np.arange(-s[2], s[2] + 1)
    M = np.transpose(np.meshgrid(m1, m2, m3)).reshape(-1, 3)
    return M[~np.all(M == 0, axis=1)] # remove zero-term

if __name__ == '__main__':
    main()