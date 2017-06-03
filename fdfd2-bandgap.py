#-*- coding: utf-8 -*-

"""Plots the band diagram of a infinite periodic arrangement of layers of different permittivity.
Uses an FDFD solver. Adjusted can be the dielectric constants, the ratio of the layer thickness, absolute size of the
layers, number of modes and number of grid nodes. """

import numpy as np
import scipy.sparse.linalg as splinalg
import matplotlib.pyplot as plt

# --- Constants --- #

n = 200                                     # number of grid nodes, should be below 300 (time)
r = 0.5                                     # ratio of layer thickness
dx = 1. / n                                 # step size, domain size = 1
e1 = 13.                                    # permittivity in the first layer
e2 = 1.                                     # permittivity in the second layer
a = 1.                                      # size of one layer
modes = 6                                   # number of modes

# --- Matrices --- #

e = np.zeros((n, n))                        # empty permittivity tensor

for i in range(0, int(r*n)):                # fills matrix with appropriate values
    e[i, i] = 1/e1

for i in range(int(r*n), n):
    e[i, i] = 1/e2

Mf = np.zeros((n, n), dtype=np.complex)     # empty matrices
Mb = np.zeros((n, n), dtype=np.complex)

for i in range(0, n):
    Mf[i, i] = -(1./dx)
    Mb[i, i] = (1./dx)

for i in range(0, n-1):
    Mf[i, i+1] = (1./dx)
    Mb[i+1, i] = -(1./dx)

# --- Solver --- #

kk0 = np.linspace(-np.pi, np.pi, 102)       # k0 vector, has to be even
k = np.zeros((modes, len(kk0)), dtype=np.complex)                       # empty k vector

for i in range(len(kk0)):
    k0 = kk0[i]
    Mf[n - 1, 0] = np.cos(k0*a)/dx          # Bloch conditions
    Mb[0, n - 1] = -np.cos(k0*a)/dx
    q = -e.dot(Mb).dot(Mf)                  # completely assembled matrix
    kguess = k0 / np.sqrt((r * e1 + e2 * (1 - r)))                      # starting vector for guessing
    k2, v = splinalg.eigs(q, k=modes, sigma=kguess ** 2)                # solver
    k[:, i] = np.sqrt(k2)

# --- Plot commands --- #

for i in range(modes):                      # plots each mode
    plt.plot(kk0 / (2 * np.pi), np.real(k[i, :] / (2 * np.pi)), '-')

plt.xlabel("k")                             # labels and plot area
plt.ylabel("$\omega$")
plt.xlim([-0.5, 0.5])
plt.ylim([0.0, 1.3])
plt.show()