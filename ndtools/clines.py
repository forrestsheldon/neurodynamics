"""
Tools for finding nullclines, crossings of curves, and calculating Jacobians
numerically.

Written by Jeffrey Bush (jeff@coderforlife.com) 2010
Adapted from UCSD Neurodynamics MATLAB clines package 
"""

import pylab as plt
import scipy as sp
import scipy.linalg as lin
from scipy.interpolate import griddata # only available in SciPy 0.9 and later

def getNullcline(d, x, y):
    """X,Y = getNullcline(d, x, y)
    getNullcline takes a matrix, the x and y scales, and returns the x and y
    values of the null-cline.
    
    Parameters:
      d  : A matrix approximating a two-dimensional function.
      x  : The x-values for the columns of f.
      y  : The y-values for the rows of f.
    
    Returns:
     X,Y : The x and y values of the null-cline.
    """
    h = plt.figure() # a dummy figure to draw the contours on
    cs = plt.contour(x, y, d, levels=[0]) # creates a contour plot with a line at 0
    paths = cs.collections[0].get_paths() # get the path of the null-cline
    xy = paths[0].vertices # get the verticies of the path
    plt.close(h) # close the dummy figure
    return xy[:,0], xy[:,1]


def getCrossings(x1,y1, x2,y2):
    """x,y = getCrossings(x1,y1,x2,y2)
    
    getCrossings approximates the places where two functions cross. x and y are
    the coordinates of those crossing points.
    """
    x, y = _intersections(x1,y1, x2,y2)
    return x, y

def _intersections(x1,y1, x2,y2):
    """X0,Y0 = intersections(X1,Y1,X2,Y2)
    INTERSECTIONS Intersections of curves.
      Computes the (x,y) locations where two curves intersect.  The curves
      can be broken with NaNs or have vertical segments.
    
    Example:
      [X0,Y0] = intersections(X1,Y1,X2,Y2);
    
    where X1 and Y1 are equal-length vectors of at least two points and
    represent curve 1.  Similarly, X2 and Y2 represent curve 2.
    X0 and Y0 are column vectors containing the points at which the two
    curves intersect.

    The algorithm can return two additional vectors that indicate which
    segment pairs contain intersections and where they are:

      [X0,Y0,I,J] = intersections(X1,Y1,X2,Y2);
    
    For each element of the vector I, I(k) = (segment number of (X1,Y1)) +
    (how far along this segment the intersection is).  For example, if I(k) =
    45.25 then the intersection lies a quarter of the way between the line
    segment connecting (X1(45),Y1(45)) and (X1(46),Y1(46)).  Similarly for
    the vector J and the segments in (X2,Y2).

    Version: 1.10, 25 February 2008
    Converted to Python October 2010 by Jeffrey Bush jeff@coderforlife.com
    Author:  Douglas M. Schwarz
    Email:   dmschwarz=ieee*org, dmschwarz=urgrad*rochester*edu
    Real_email = regexprep(Email,{'=','*'},{'@','.'})

    Theory of operation:
      Given two line segments, L1 and L2,
    
      L1 endpoints:  (x1(1),y1(1)) and (x1(2),y1(2))
      L2 endpoints:  (x2(1),y2(1)) and (x2(2),y2(2))
    
    we can write four equations with four unknowns and then solve them.  The
    four unknowns are t1, t2, x0 and y0, where (x0,y0) is the intersection of
    L1 and L2, t1 is the distance from the starting point of L1 to the
    intersection relative to the length of L1 and t2 is the distance from the
    starting point of L2 to the intersection relative to the length of L2.
    
    So, the four equations are
    
       (x1(2) - x1(1))*t1 = x0 - x1(1)
       (x2(2) - x2(1))*t2 = x0 - x2(1)
       (y1(2) - y1(1))*t1 = y0 - y1(1)
       (y2(2) - y2(1))*t2 = y0 - y2(1)
    
    Rearranging and writing in matrix form,
    
      [x1(2)-x1(1)       0       -1   0;      [t1;      [-x1(1);
            0       x2(2)-x2(1)  -1   0;   *   t2;   =   -x2(1);
       y1(2)-y1(1)       0        0  -1;       x0;       -y1(1);
            0       y2(2)-y2(1)   0  -1]       y0]       -y2(1)]
    
    Let's call that A*T = B.  We can solve for T with T = A\B.
    
    Once we have our solution we just have to look at t1 and t2 to determine
    whether L1 and L2 intersect.  If 0 <= t1 < 1 and 0 <= t2 < 1 then the two
    line segments cross and we can include (x0,y0) in the output.
    
    In principle, we have to perform this computation on every pair of line
    segments in the input data.  This can be quite a large number of pairs so
    we will reduce it by doing a simple preliminary check to eliminate line
    segment pairs that could not possibly cross.  The check is to look at the
    smallest enclosing rectangles (with sides parallel to the axes) for each
    line segment pair and see if they overlap.  If they do then we have to
    compute t1 and t2 (via the A\B computation) to see if the line segments
    cross, but if they don't then the line segments cannot cross.  In a
    typical application, this technique will eliminate most of the potential
    line segment pairs.
    """

    # x1 and y1 must be vectors with same number of points (at least 2).
    if sp.sum(sp.size(x1) > 1) != 1 or sp.sum(sp.size(y1) > 1) != 1 or len(x1) != len(y1):
        raise ValueError('X1 and Y1 must be equal-length vectors of at least 2 points.')
    # x2 and y2 must be vectors with same number of points (at least 2).
    if sp.sum(sp.size(x2) > 1) != 1 or sp.sum(sp.size(y2) > 1) != 1 or len(x2) != len(y2):
        raise ValueError('X2 and Y2 must be equal-length vectors of at least 2 points.')

    # Compute number of line segments in each curve and some differences we'll
    # need later.
    n1 = len(x1) - 1
    n2 = len(x2) - 1
    xy1 = sp.column_stack((x1, y1))
    xy2 = sp.column_stack((x2, y2))
    dxy1 = sp.diff(xy1, axis=0)
    dxy2 = sp.diff(xy2, axis=0)

    # Determine the combinations of i and j where the rectangle enclosing the
    # i'th line segment of curve 1 overlaps with the rectangle enclosing the
    # j'th line segment of curve 2.
    i, j = sp.nonzero(sp.logical_and(sp.logical_and(sp.logical_and(
        sp.tile(sp.minimum(x1[0:-1],x1[1:]), (n2,1)).T <= sp.tile(sp.maximum(x2[0:-1],x2[1:]), (n1,1)), 
	sp.tile(sp.maximum(x1[0:-1],x1[1:]), (n2,1)).T >= sp.tile(sp.minimum(x2[0:-1],x2[1:]), (n1,1))),
	sp.tile(sp.minimum(y1[0:-1],y1[1:]), (n2,1)).T <= sp.tile(sp.maximum(y2[0:-1],y2[1:]), (n1,1))),
	sp.tile(sp.maximum(y1[0:-1],y1[1:]), (n2,1)).T >= sp.tile(sp.minimum(y2[0:-1],y2[1:]), (n1,1)))
        )
    i = sp.copy(i) # make the arrays writable
    j = sp.copy(j)

    # Find segments pairs which have at least one vertex = NaN and remove them.
    # This line is a fast way of finding such segment pairs.  We take
    # advantage of the fact that NaNs propagate through calculations, in
    # particular subtraction (in the calculation of dxy1 and dxy2, which we
    # need anyway) and addition.
    remove = sp.isnan(sp.sum(dxy1[i,:] + dxy2[j,:], axis=1))
    i[remove] = []
    j[remove] = []

    # Initialize matrices.  We'll put the T's and B's in matrices and use them
    # one column at a time.  AA is a 3-D extension of A where we'll use one
    # plane at a time.
    n = len(i)
    T = sp.zeros((4, n))
    AA = sp.zeros((4, 4, n))
    AA[[0, 1],2,:] = -1
    AA[[2, 3],3,:] = -1
    AA[[0, 2],0,:] = dxy1[i,:].T
    AA[[1, 3],1,:] = dxy2[j,:].T
    B = -sp.array([x1[i], x2[j], y1[i], y2[j]])

    # Loop through possibilities.  Trap singularity warning and then use
    # lastwarn to see if that plane of AA is near singular.  Process any such
    # segment pairs to determine if they are colinear (overlap) or merely
    # parallel.  That test consists of checking to see if one of the endpoints
    # of the curve 2 segment lies on the curve 1 segment.  This is done by
    # checking the cross product
    #
    #   (x1(2),y1(2)) - (x1(1),y1(1)) x (x2(2),y2(2)) - (x1(1),y1(1)).
    #
    # If this is close to zero then the segments overlap.
    for k in sp.arange(n):
        L,U = lin.lu(AA[:,:,k], True)
        T[:,k] = lin.solve(U, lin.solve(L, B[:,k]))

    # Find where t1 and t2 are between 0 and 1 and return the corresponding
    # x0 and y0 values.
    in_range = sp.logical_and(sp.logical_and(sp.logical_and(T[0,:]>=0, T[1,:]>=0), T[0,:]<1), T[1,:]<1)
    x0 = T[2,in_range].T
    y0 = T[3,in_range].T

    return x0, y0


def getJacobian(x,y,f,g,x0,y0):
    """J = getJacobian(x,y,f,g,x0,y0)
    getJacobian takes a grid of derivative values and a point in phase
    space, and returns the Jacobian around that point.

    Parameters:
      x   : The x-values of f and g.
      y   : The y-values of f and g.
      f   : A matrix approximating the derivative wrt x.
      g   : A matrix approximating the derivative wrt y.
      x0  : The x-value for the point.
      y0  : The y-value for the point.

    Returns Jacobian:
    J(1,1): Derivative of f wrt x evaluated at x0, y0.
    J(1,2): Derivative of f wrt y evaluated at x0, y0.
    J(2,1): Derivative of g wrt x evaluated at x0, y0.
    J(2,2): Derivative of g wrt y evaluated at x0, y0.
    """
    dx = sp.gradient(x)[1] # the derivative in the X direction
    dy = sp.gradient(y)[0] # the derivative in the Y direction
    dfy, dfx = sp.gradient(f) # the derivatives of f in the X and Y directions
    dgy, dgx = sp.gradient(g) # the derivatives of g in the X and Y directions

    # Now we need to get the values at the fixed point. We have to interpolate
    # the data from what we have.
    points = (x.flatten(), y.flatten())
    point = (x0, y0)
    dx0   = griddata(points, dx.flatten(),  point)
    dy0   = griddata(points, dy.flatten(),  point)
    dfdx0 = griddata(points, dfx.flatten(), point)
    dfdy0 = griddata(points, dfy.flatten(), point)
    dgdx0 = griddata(points, dgx.flatten(), point)
    dgdy0 = griddata(points, dgy.flatten(), point)

    #X, Y = x.flatten(), y.flatten()
    #xi,yi = plt.meshgrid([x0-1, x0, x0+1], [y0-1, y0, y0+1])
    #dx0   = griddata(X, Y, dx.flatten(), xi,yi)[1][1]
    #dy0   = griddata(X, Y, dy.flatten(), xi,yi)[1][1]
    #dfdx0 = griddata(X, Y, dfx.flatten(),xi,yi)[1][1]
    #dfdy0 = griddata(X, Y, dfy.flatten(),xi,yi)[1][1]
    #dgdx0 = griddata(X, Y, dgx.flatten(),xi,yi)[1][1]
    #dgdy0 = griddata(X, Y, dgy.flatten(),xi,yi)[1][1]

    return sp.array([[dfdx0/dx0, dfdy0/dy0], [dgdx0/dx0, dgdy0/dy0]])
