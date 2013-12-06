# calculates autocorrelation as a function of time for the randomly connected
# network model

import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import odeint

# seed random number generator
np.random.seed(28031987)

N = 200  # number of network elements
# generate instance of random matrix
J_orig = np.random.randn(N,N)

# create array of desired Jsig vals
Jvec = (1.05, 1.1, 1.2, 1.25, 1.26, 1.266, 1.2668, 1.27, 1.3, 1.4)

# equations of motion
def dxdt(x, t, gJ, J):
    return -x + np.einsum('ij,j', J, np.tanh(gJ*x))

for Jval in Jvec:
    # rescale variance of matrix elements
    J2 = Jval*Jval
    Jvar = J2/N
    J = np.sqrt(Jvar) * J_orig

    for i in range(N):
        J[i,i] = 0.0
    
    # set gain parameter
    g = 1.0
    gJ = g * np.sqrt(J2)
    
    # randomly draw initial state
    x = np.array(np.random.randn(N))
    
    # integrate out the transient behavior
    tstart = time.time()
    t0 = 0.0
    tf = 6000.0
    dt = 0.1
    times = np.arange(t0, tf+dt, dt)
    
    sol = odeint(dxdt, x, times, (gJ,J))
    x = sol[-1]
    print time.time()-tstart, ' s'
    
    # now generate time series data
    tstart = time.time()
    t0 = 0.0
    tf = 10000.0
    dt = 0.1
    times = np.arange(t0, tf+dt, dt)
    
    sol = odeint(dxdt, x, times, (gJ,J))
    x = sol[-1]
    print time.time()-tstart, ' s'
    
    # plot phase space trajectories
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(sol[:,0], sol[:,1])
    ax.set_xlabel(r'$x_1$')
    ax.set_ylabel(r'$x_2$')
    fig.savefig('figs/J_' + str(Jval) + '.pdf')
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(sol.T[0],sol.T[1],sol.T[2])
    ax.set_xlabel(r'$x_1$')
    ax.set_ylabel(r'$x_2$')
    ax.set_zlabel(r'$x_3$')
    fig.savefig('figs/J_' + str(Jval) + '_3d.pdf')
    
    # now calculate and plot time-correlation function
    tstart = time.time()
    corr_vec = np.correlate(sol[:,0], sol[:,0], mode='full')
    corr_vec /= np.sqrt(np.einsum('i,i', corr_vec, corr_vec))
    print time.time()-tstart, ' s'
    
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(corr_vec)
    fig.savefig('figs/tcorr_J_' + str(Jval) + '.pdf')
    
    plt.close(fig)
    










