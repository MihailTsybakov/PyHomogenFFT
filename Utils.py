"""
    Helper funcs
"""

import numpy as np
import pyvista as pv

def save_3d_array_as_vtk(array_3d, filename, origin=(0, 0, 0), spacing=(1, 1, 1)):
    """
    Save a 3D NumPy array to a .vtk file using PyVista's StructuredGrid.
    
    Parameters
    ----------
    array_3d : numpy.ndarray
        3D NumPy array of shape (nx, ny, nz) containing the scalar values.
    filename : str
        The output filename (e.g. 'output.vtk' or 'output.vti').
    origin : tuple of float, optional
        The (x, y, z) origin of the dataset.
    spacing : tuple of float, optional
        The spacing between points along each axis.
    """
    
    # Ensure the array is 3D
    if array_3d.ndim != 3:
        raise ValueError("Input array must be 3D.")
    
    nx, ny, nz = array_3d.shape
    
    x = np.linspace(origin[0], origin[0] + spacing[0]*(nx-1), nx)
    y = np.linspace(origin[1], origin[1] + spacing[1]*(ny-1), ny)
    z = np.linspace(origin[2], origin[2] + spacing[2]*(nz-1), nz)
    
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    
    grid = pv.StructuredGrid(xx, yy, zz)
    
    grid["Data"] = array_3d.ravel(order='F')
    
    grid.save(filename)


def add_sphere(model, cx, cy, cz, r):
    """
    Adds spherical inclusion into given matrix

    Parameters:
        model (3D np.array) - RVE voxel model 
        cx,cy,cz (int) - inclusion center
        r (float) - inclusion radius

    Returns:
        - Updates model in-place
    """

    Nx, Ny, Nz = model.shape 
    X, Y, Z = np.meshgrid(np.arange(Nx), np.arange(Ny), np.arange(Nz), indexing='ij')
    R2      = (X - cx)**2 + (Y - cy)**2 + (Z - cz)**2

    model[R2 <= r**2] = 1