# -*- coding: utf-8 -*-

#
# Generate the 1pi molecular orbitals of CO at the HF/sto3g level of theory
#

from pyqint import PyQInt, HF, MoleculeBuilder
import numpy as np

def main():
    mol = MoleculeBuilder().from_name('CO')
    res = HF().rhf(mol, 'sto3g')
    
    # construct the wave functions in a cubic unit cell of 10x10x10 with
    # a sampling of 64 grid points per Cartesian direction
    mo5 = plot_wavefunction(res['cgfs'], res['orbc'][:,4])
    mo6 = plot_wavefunction(res['cgfs'], res['orbc'][:,5])
    
    # store the wave functions
    np.save('mo5.npy', mo5)
    np.save('mo6.npy', mo6)
    
    # calculate the kinetic energy
    Ekin5 = res['orbc'][:,4] @ res['kinetic'] @ res['orbc'][:,4]
    Ekin6 = res['orbc'][:,5] @ res['kinetic'] @ res['orbc'][:,5]
    print(Ekin5)
    print(Ekin6)

def plot_wavefunction(cgfs, coeff):
    # build integrator
    integrator = PyQInt()

    # build grid
    sz = 64
    grid = integrator.build_rectgrid3d(-5, 5, sz)
    res = integrator.plot_wavefunction(grid, coeff, cgfs)

    return res

if __name__ == '__main__':
    main()

