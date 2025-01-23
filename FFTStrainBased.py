"""
Spectral FFT-Driven Strain-Based Homogenization Solver
H. Moulinec & P. Suquet, 1998
"""


import numpy as np 


""" 
 Tensor operations
"""
trans2 = lambda A2   : np.einsum('ijxyz          ->jixyz  ',A2   )
ddot22 = lambda A2,B2: np.einsum('ijxyz  ,jixyz  ->xyz    ',A2,B2)
ddot42 = lambda A4,B2: np.einsum('ijklxyz,lkxyz  ->ijxyz  ',A4,B2)
ddot44 = lambda A4,B4: np.einsum('ijklxyz,lkmnxyz->ijmnxyz',A4,B4)
dot11  = lambda A1,B1: np.einsum('ixyz   ,ixyz   ->xyz    ',A1,B1)
dot22  = lambda A2,B2: np.einsum('ijxyz  ,jkxyz  ->ikxyz  ',A2,B2)
dot24  = lambda A2,B4: np.einsum('ijxyz  ,jkmnxyz->ikmnxyz',A2,B4)
dot42  = lambda A4,B2: np.einsum('ijklxyz,lmxyz  ->ijkmxyz',A4,B2)
dyad22 = lambda A2,B2: np.einsum('ijxyz  ,klxyz  ->ijklxyz',A2,B2)


def BasicScheme(E, nu, DE, reltol, maxiter):
    """
    Strain-Based FFT Solver for prescribed strain homogenizer

    Parameters:
        np.ndarray: E  - Young modulus field
        np.ndarray: nu - Poisson ratio field
        np.ndarray: DE - prescribed average strain 
        float: reltol  - convergence eps
        int: maxiter

    Returns:
        np.ndarray: sig - stress tensor field
        np.ndarray: eps - strain tensor field
        np.ndarray: err_hist - convergence history
    """

    Nx, Ny, Nz = E.shape
    shape      = (Nx, Ny, Nz)

    if (Nx % 2 == 0 or Ny % 2 == 0 or Nz % 2 == 0):
        raise RuntimeError("Error: domain shape should be odd")


    """
     Needed identities
    """
    i      = np.eye(3).astype(np.float32)
    I      = np.einsum('ij,xyz'           ,                  i   ,np.ones([Nx, Ny, Nz])).astype(np.float32)
    I4     = np.einsum('ijkl,xyz->ijklxyz',np.einsum('il,jk',i,i),np.ones([Nx, Ny, Nz])).astype(np.float32)
    I4rt   = np.einsum('ijkl,xyz->ijklxyz',np.einsum('ik,jl',i,i),np.ones([Nx, Ny, Nz])).astype(np.float32)
    I4s    = ((I4+I4rt)/2).astype(np.float32)
    II     = dyad22(I,I).astype(np.float32)
    I4d = (I4s - II/3).astype(np.float32)


    """ FFT Routine with shift """
    fft  = lambda x: np.fft.fftshift(np.fft.fftn (np.fft.ifftshift(x),[Nx,Ny,Nz]))
    ifft = lambda x: np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(x),[Nx,Ny,Nz]))


    """ Lame parameters in tensor form"""
    lame_lambda = (E*nu / ((1+nu) * (1-2*nu))).astype(np.float32)
    lame_mu = (E / (2*(1+nu))).astype(np.float32)
    K = (E / (3*(1 - 2*nu))).astype(np.float32)


    """ Reference media properties """
    lambda_0 = 0.5*(np.min(lame_lambda) + np.max(lame_lambda))
    mu_0 = 0.5*(np.min(lame_mu) + np.max(lame_mu))


    """ 4th order stiffness tensor """
    C4 = (K*II + 2*lame_mu*I4d).astype(np.float32)


    """ Frequency domain discretization """
    freq_x = np.arange(-(Nx-1)/2, +(Nx+1)/2)
    freq_y = np.arange(-(Ny-1)/2, +(Ny+1)/2)
    freq_z = np.arange(-(Nz-1)/2, +(Nz+1)/2)

    X = np.zeros((3, Nx,Ny,Nz))
    Freq = np.zeros_like(X)

    X[0], X[1], X[2] = np.mgrid[:Nx, :Ny, :Nz]

    Freq[0] = freq_x[X[0].astype(np.int32)]
    Freq[1] = freq_y[X[1].astype(np.int32)]
    Freq[2] = freq_z[X[2].astype(np.int32)]

    Q       = dot11(Freq, Freq)
    Z       = Q==0
    Q[Z]    = 1
    norm    = 1/Q
    norm[Z] = 0

    delta_ij = lambda i,j: float(i == j)


    """ Green operator assembly """
    G0 = np.zeros((3,3,3,3, Nx,Ny,Nz)).astype(np.float32)
    for k in range(3):
        for h in range(3):
            for i in range(3):
                for j in range(3):
                    term_1 = (delta_ij(k,i)*Freq[h]*Freq[j] + delta_ij(h,i)*Freq[k]*Freq[j] + delta_ij(k,j)*Freq[h]*Freq[i] + delta_ij(h,j)*Freq[i]*Freq[k]) / (4*mu_0)
                    term_2 = (lambda_0 + mu_0) / (mu_0*(lambda_0 + 2*mu_0))*Freq[k]*Freq[h]*Freq[i]*Freq[j]
                    
                    G0[k,h,i,j] = term_1*norm - term_2*norm**2


    """ Tensor initialization """
    sig = np.zeros((3,3, Nx,Ny,Nz)).astype(np.float32)
    eps = np.zeros((3,3, Nx,Ny,Nz)).astype(np.float32)

    eps = DE.copy()
    sig = ddot42(C4, eps)


    """ Fixed-point iteration cycle """
    iter_      = 0
    delta_hist = []
    eps_prev   = DE.copy()

    while True:

        sig_freq = fft(sig)
        eps_freq = fft(eps)

        eps_freq = eps_freq - ddot42(G0, sig_freq)

        eps = np.real(ifft(eps_freq))
        sig = ddot42(C4, eps)

        iter_ += 1

        eps_div = eps.copy()
        eps_div[eps == 0] = 1.0
        rel_strain_change = np.max(np.abs(eps_prev - eps) / eps_div)

        print(f'Iter {iter_}  |  strain change = {rel_strain_change:.3e}  |  eps = {reltol:.3e}')

        if (rel_strain_change <= reltol):
            print(f'Converged')
            break 

        if (iter_ >= maxiter):
            print(f'Maxiter reached, exit')
            break

        eps_prev = eps.copy()


    return sig, eps, delta_hist

