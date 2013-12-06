#from mpl_toolkits.mplot3d import Axes3D

import time, cPickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

Nelem = 1000  # number of network elements
Nensm = 100

# rescale variance of matrix elements
J2 = 1.2*1.2
Jvar = J2/Nelem

# equations of motion
g = 1.0
def dxdt(x, t, g, J):
        return -x + np.einsum('ij,j', J, np.tanh(g*x))

for nrun in range(Nensm):
    print str(nrun)
    
    # get state of random number generator, pickle in case transient not run out
    rand_state = np.random.get_state()
    f = open('data_testloop/randstate_J_' + str(np.sqrt(J2)) + '_run' + str(nrun) + '.pickle', 'wb')
    cPickle.dump(rand_state, f, protocol=2)
    f.close()
    
    # generate instance of random matrix
    J_orig = np.random.randn(Nelem,Nelem)
    J = np.sqrt(Jvar) * J_orig
    
    for i in range(Nelem):
        J[i,i] = 0.0
    
    # randomly draw initial state
    x = np.array(np.random.randn(Nelem))
    # instead use x on the limit cycle found in one set of experiments
    #x = np.load('limitcycle_x0.npy')
    
    # let the transient run out
    tstart = time.time()  # timing purposes
    print 'Running out transient...'
    t0 = 0.0
    tf = 3000.0  # assumed transient time
    dt = 1.0
    times = np.arange(t0, tf+dt, dt)
    
    sol = odeint(dxdt, x, times, (g,J))
    x = sol[-1]
    print time.time() - tstart, ' s'
    
    # now run for a little while to collect data
    tstart = time.time()  # timing purposes
    print 'Running data collection period...'
    t0 = 0.0
    tf = 4000.0
    dt = 0.1
    times = np.arange(t0, tf+dt, dt)
    
    sol = odeint(dxdt, x, times, (g,J))
    x = sol[-1]
    print time.time() - tstart, ' s'
    
    # save time series figs, also phase space plots
    """
    fig = plt.figure()
    ax = fig.add_subplot(5,1,1)
    ax.plot(times, sol[:,0])
    ax = fig.add_subplot(5,1,2)
    ax.plot(times, sol[:,1])
    ax = fig.add_subplot(5,1,3)
    ax.plot(times, sol[:,2])
    ax = fig.add_subplot(5,1,4)
    ax.plot(times, sol[:,3])
    ax = fig.add_subplot(5,1,5)
    ax.plot(times, sol[:,4])
    fname = 'figs_testloop/ts_J_' + str(np.sqrt(J2)) + '_run' + str(nrun) + '.pdf'
    fig.savefig(fname)
    plt.close(fig)
    """
    
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(sol[:,0],sol[:,1])
    fname = 'figs_testloop/ps_J_' + str(np.sqrt(J2)) + '_run' + str(nrun) + '.pdf'
    fig.savefig(fname)
    plt.close(fig)
    
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(sol.T[0],sol.T[1],sol.T[2])
    fig.savefig('./figs_testloop/ps_J_' + str(np.sqrt(J2)) + '_run' + str(nrun) + '_3d.pdf')
    """




