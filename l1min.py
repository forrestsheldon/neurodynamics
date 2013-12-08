#
# l1min: general purpose solver for real and complex l1 minimization problems
#
# Author: Mohammadreza Mahmudimanesh
#         Technische Universitaet Darmstadt, Germany
#         http://www.deeds.informatik.tu-darmstadt.de/reza
#
# Version 0.9 (beta)  15 Apr 2011
#


from cvxopt import matrix,solvers


# l1m(A, b, e) solves the following convex optimization problem using CVXOPT
#
#       minimize |x|_1   subject to   |Ax-b|_2 <= e
# 
# where A is an m by n real or complex matrix, b is a real or complex vector
# of size m and e is a nonnegative real number.
#
# Default value for argument e is zero. Therefore, by default the above
# optimzation problem is simplified to the following problem with equality
# constraint.
#
#       minimize |x|_1   subject to   Ax = b

def l1m(A, b, e=0):
	if e<0:
		print 'l1m: e must be nonnegative'
		return None
	m = A.size[0]
	n = A.size[1]
	if b.size[0]!=m:
		print "number of A's rows must be equal to the size of b"
		return None
	if A.typecode=='z' or b.typecode=='z':
		return cl1m_socp(A, m, n, b, e)
	else:
		if e>0:
			return rl1m_socp(A, m, n, b, e)
		else:
			return rl1m_lp(A, m, n, b)


# rl1m_lp: solves the following linear program
#
#       minimize |x|_1   subject to   Ax = b
#
# where A is an m by n real matrix and b is a real vector of size m.

def rl1m_lp(A, m, n, b):
	c = matrix(0.0, (2*n,1))
	c[n:2*n] = 1.0
	Aa = matrix(0.0, (m,2*n))
	Aa[:,0:n] = A		
	G = matrix(0.0, (3*n,2*n))
	for i in range(n):
		G[i,i] = 1.0
		G[n+i,i] = G[i,n+i] = G[n+i,n+i] = G[2*n+i,n+i] = -1.0
	h = matrix(0.0, (3*n,1))
	sol = solvers.lp(c, G, h, Aa, b)
	s = sol['x']
	return s[0:n]


# rl1m_socp: solves the following second order cone program
#
#       minimize |x|_1   subject to   |Ax-b|_2 <= e
#
# where A is an m by n real matrix, b is a real vector of size m
# and e is a nonnegative real number.

def rl1m_socp(A, m, n, b, e):
	c = matrix(0.0, (2*n,1))
	c[n:2*n] = 1.0
	Gl = matrix(0.0, (3*n,2*n))
	for i in range(n):
		Gl[i,i] = 1.0
		Gl[n+i,i] = Gl[i,n+i] = Gl[n+i,n+i] = Gl[2*n+i,n+i] = -1.0
	hl = matrix(0.0, (3*n,1))
	Gq = []
	hq = []
	G = matrix(0.0, (m+1,2*n))
	G[1:m+1,0:n] = A
	Gq.append(G)
	h = matrix(0.0, (m+1,1))
	h[0] = e
	h[1:m+1] = b
	hq.append(h)
	sol = solvers.socp(c, Gl, hl, Gq, hq)
	s = sol['x']
	return s[0:n]


# cl1m_socp: solves the following second order cone program
#
#       minimize |x|_1   subject to   |Ax-b|_2 <= e
#
# where A is an m by n complex matrix, b is a complex vector of size m
# and e is a nonnegative real number. When e is zero, the above problem
# is transformed to
#
#       minimize |x|_1   subject to   Ax = b

def cl1m_socp(A, m, n, b, e):
	c = matrix(0.0, (3*n,1))
	c[range(0,3*n,3)] = 1.0
	Ga=[]
	ha=[]
	for i in range(n):
	    G = matrix(0.0, (3,3*n))
	    G[0,3*i] = G[1,3*i+1] = G[2,3*i+2] = -1.0
	    Ga.append(G)
	    ha.append(matrix(0.0,(3,1)))
	sol = None
	Aa = matrix(0.0, (2*m,3*n))
	Aa[0:m,range(1,3*n,3)] = A.real()
	Aa[0:m,range(2,3*n,3)] = -A.imag()
	Aa[m:2*m,range(1,3*n,3)] = A.imag()
	Aa[m:2*m,range(2,3*n,3)] = A.real()
	y = matrix(0.0,(2*m,1))
	y[0:m] = b.real()
	y[m:2*m] = b.imag()
	if e==0:
		sol = solvers.socp(c, Gq=Ga, hq=ha, A=Aa, b=y)
	else:
		G = matrix(0.0, (2*m+1,3*n))
		G[1:2*m+1,:] = Aa
		Ga.append(G)
		h = matrix(0.0, (2*m+1,1))
		h[0] = e
		h[1:2*m+1] = y
		ha.append(h)
		sol = solvers.socp(c, Gq=Ga, hq=ha)
	s = sol['x']
	z = matrix(complex(0,0),(n,1))
	for i in range(n): z[i] = complex(s[3*i+1], s[3*i+2])
	return z


__all__ = ['l1m']
