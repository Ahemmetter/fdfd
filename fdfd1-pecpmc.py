#-*- coding: utf-8 -*-

"""This program simulates the behaviour of a PEC-PMC cavity (perfect electric conductor,
perfect magnetic conductor). The solver finds the modes (standing waves) and displays the field
distribution inside the cavity. Calculation is done with an FDFD solver."""

import numpy as np
import scipy.sparse.linalg as splinalg
import matplotlib.pyplot as plt

# --- Constants --- #

n = 50                          # number of grid nodes
dx = 1. / n                     # step size
e = 1.                           # permittivity (1 for vacuum)
m = 1.                           # permeability (1 for vacuum)
l = 1. / (dx**2 * m * e)         # step factor for matrix
kstart = 2*np.pi                # wave vector target to find resonance wavelength
modes = 6                       # number of modes that are being calculated

# --- Matrices --- #

Mf = np.zeros((n, n))           # empty matrices
Mb = np.zeros((n, n))

for i in range(0, n):           # fills sparse matrices
    Mf[i, i] = 1
    Mb[i, i] = 1

for i in range(0, n-1):
    Mf[i, i+1] = -1
    Mb[i+1, i] = -1

M = l * np.dot(Mb, Mf)              # dot multiplication to obtain M

# --- Solver --- #

k, v = splinalg.eigs(M, modes, sigma=kstart)                    # solve for eigenvalues and vectors
vi = np.linspace(0, modes-1, modes, dtype=np.int)               # indexes eigenvectors
k2, vi = (list(t) for t in zip(*sorted(zip(np.sqrt(k), vi))))   # sorts tuples of eigenvalues and -vectors by size
kl = np.sort(np.sqrt(k2))                                       # sort
lam = 2*np.pi/np.real(kl)                                       # calculates wavelength from k-vectors

# --- Plot commands --- #

plt.figure(1)
modearray = (2*np.linspace(0, modes-1, modes))+1.             # simplification for analytical result
plt.plot(np.linspace(1, modes, modes), lam, '-')              # plots numerical solution
plt.plot(np.linspace(1, modes, modes), 4/modearray, '*')      # plots analytical solution

f, ax = plt.subplots(6,1, sharex=True, sharey=True)
for n in np.arange(v.shape[1]):
    ax[n].plot(np.abs(v[:, vi[n]]), '-')
plt.show()
