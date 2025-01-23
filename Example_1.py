"""
    Simple synthetic RVE FFT homogenization example
"""

from matplotlib     import pyplot as plt 
from FFTStrainBased import *
from Utils          import *


# Domain shape 
nx, ny, nz = 61, 61, 61

# Material props
E0, E1 = 10000, 20000
nu0, nu1 = 0.3, 0.350

# Material map
model = np.zeros((nx, ny, nz))
n_inclusions = 18
r            = 6
for i in range(n_inclusions):
    cx_ = np.random.randint(5, nx-5)
    cy_ = np.random.randint(5, ny-5)
    cz_ = np.random.randint(5, nz-5)
    add_sphere(model, cx_, cy_, cz_, r)


# Material arrays
E = np.ones((nx, ny, nz)) * E0
nu= np.ones((nx, ny, nz)) * nu0 
E[model == 1] = E1 
nu[model == 1] = nu1 

# Prescribed strain
DE = np.zeros((3,3, nx,ny,nz))
DE[2, 2, :, :, :] = 1e-3

# Solving 
sig, eps, hist = BasicScheme(E, nu, DE, reltol = 1e-2, maxiter = 20)


# Output to VTK
save_3d_array_as_vtk(sig[0,0], "Test.vtk")