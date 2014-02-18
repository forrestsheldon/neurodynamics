"""
calc_le.py

Created by Paul Rozdeba on December 5, 2013

For a dynamical system, takes an initial separation vector u_0 and the 
(numerically evaluated) "Jacobian" matrix J_ij(t) = dx_i(t0+t)/dx_j(t0), 
along with the mapping F(x(t)).  It evolves the state forward in time with the 
mapping, and the Jacobian forward with the variational equation
    dJ_ij(t)/dt = dF_i(x(t0+t))/dx(t0) + J_ij(t)

The calculation is largely based off M. Sandri, "Numerical calculation of 
Lyapunov exponents".
"""

import numpy as np
from scipy.integrate import odeint

def dJdt(Jrow, t, Df):
    """
    Returns row of dJ_{ij}(t0+t)/dt, according to the variational equation.
    """
    
    return np.einsum('ij,i', Df, Jrow)

def local_LE(times, f, Df, x0_a, x0_b, fargs=None, Dfargs=None):
    """
    Calculates a time series of local Lyapunov exponents.  At each point in 
    time, returns the largest LE.
    
    Arguments:
        times: times at which to calculate local LE.
        f: dynamical system dx_i(t)/dt.
        Df: matrix-value function, partial derivatives df_i(x(t0+t))/dx_j, 
            i.e. take partials and evaluate at t0+t.  Df should take the 
            current position x plus parameters as arguments
        x0_a, x0_b: initial positions, separation should be small (these are 
            used to calculate the initial separation vector).  x0_a is used, 
            by default, as the initial position for the variational equation.
        *fargs: additional arguments to pass to f
        *Dfargs: additional arguments to pass to Df
    """
    
    # initialization
    ND = x0_a.shape[0]  # extract dimensionality of system
    LLE_vec = np.zeros(shape=(len(times),), dtype='float')  # for storing LLE's
    x = x0_a  # initialize position vector
    u0 = x0_b - x0_a  # initial separation vector
    u = u0
    J = np.eye(ND, dtype='float')  # Jacobian at t0 is identity matrix
    
    # loop over t, save LLE at each time
    for nt in range(1,len(times)):
        for iJ in range(ND):  # new Jacobian, row by row
            J[iJ] = odeint(dJdt, J[iJ], (times[nt-1],times[nt]), (Df(x,Dfargs),))[-1]
        x = odeint(f, x, (times[nt-1],times[nt]), fargs)[-1]  # new pos
        
        # evolve separation vector forward in time
        u = np.einsum('ij,j', J,u0)
        # calculate and save LLE
        LLE_vec[nt] = np.log(np.einsum('i,i', u,u)) / (2.0*times[nt])
        # factor of 1/2 above comes from sqrt of sep vec squared, inside log
    
    return LLE_vec
    





