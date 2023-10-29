from itertools import combinations
import numpy as np

def distance(h1,h2):
    return np.linalg.norm(np.array(h1)-np.array(h2))
    
def coords_astrometry(A,B,C,D):
    
    A, B, C, D = A-A, B-A, C-A, D-A
    
    X = (B + np.array([[0,1],[-1,0]]) @ B) / 2
    Y = (B + np.array([[0,-1],[+1,0]]) @ B) / 2

    xc, yc = np.dot(C,X) / np.linalg.norm(C) / np.linalg.norm(X), np.dot(C,Y) / np.linalg.norm(C) / np.linalg.norm(Y)
    xd, yd = np.dot(D,X) / np.linalg.norm(D) / np.linalg.norm(X), np.dot(D,Y) / np.linalg.norm(D) / np.linalg.norm(Y)

    return np.array([xc,yc,xd,yd])
        
def hash_astrometry(points):
    max_distance = -1
    farthest_pair = None
    farthest_i_j = None, None
    
    for i, point1 in enumerate(points):
        for j, point2 in enumerate(points):
            if i >= j:
                continue
            dist = distance(point1, point2)
            if dist > max_distance:
                max_distance = dist
                farthest_pair = (point1, point2)
                farthest_i_j = i, j

    i, j = farthest_i_j
    
    A = np.array(points[i])
    B = np.array(points[j])
    
    k, l = [m for m in [0,1,2,3] if m not in [i,j]]
    
    C = np.array(points[k])
    D = np.array(points[l])

    xc,yc,xd,yd = coords_astrometry(A,B,C,D)

    # this is the astrometry.net invariant (under scaling, rotation and translation)

    if not xc <= xd:
        A, B = B, A
        xc,yc,xd,yd = coords_astrometry(A,B,C,D)
        
    if not xc+xd <= 1:
        C, D = D, C
        xc,yc,xd,yd = coords_astrometry(A,B,C,D)

    return xc,yc,xd,yd
    
def order(points):
    # Calculate the centroid of the points
    cx = np.mean([p[0] for p in points])
    cy = np.mean([p[1] for p in points])

    # Sort the points based on the angle with respect to the centroid
    sorted_points = sorted(points, key=lambda p: np.arctan2(p[1]-cy, p[0]-cx))

    return sorted_points

def hash_juan_old(points):
    P1,P2,P3,P4 = points     
    P1,P2,P3,P4 = order(points)
    d1,d2,d3,d4 = map(np.linalg.norm, [P2-P1,P3-P2,P4-P3,P1-P4])
    return d1,d2,d3,d4



def quad_coords_juan(A,B,C,D):

    P = (A+B)/2
    A, B, C, D = A-P,B-P,C-P,D-P
    
    X = B / np.linalg.norm(B)
    Y = np.array([[0,-1],[+1,0]]) @ X

    xa, ya = np.dot(A,X), np.dot(A,Y)
    xb, yb = np.dot(B,X), np.dot(B,Y)
    xc, yc = np.dot(C,X), np.dot(C,Y)
    xd, yd = np.dot(D,X), np.dot(D,Y)

    A = np.array([xa,ya])
    B = np.array([xb,yb])
    C = np.array([xc,yc])
    D = np.array([xd,yd])

    return A,B,C,D


def force_AB_maxdist(points):
    """ Given points = A,B,C,D,... reorders then and returns A',B',C',... A' and B' are the maximum distance points."""
    
    max_distance = -1
    farthest_i_j = None, None
    
    for i, point1 in enumerate(points):
        for j, point2 in enumerate(points):
            if i >= j:
                continue
            dist = distance(point1, point2)
            if dist > max_distance:
                max_distance = dist
                farthest_i_j = i, j

    i, j = farthest_i_j
    
    result = [points[i], points[j]]

    result.extend([points[k] for k in range(len(points)) if k not in [i,j]])
    
    return result
    

def hash_juan(points):
    A,B,C,D = points
    A,B,C,D = force_AB_maxdist([A,B,C,D])
    A,B,C,D = quad_coords_juan(A,B,C,D)

    FX = np.array([[-1,0],[0,1]])
    FY = np.array([[1,0],[0,-1]])

    M = np.identity(2)
    
    if C[0] > D[0]:
        M = FX @ M

    if C[1] > D[1]:
        M = FY @ M

    B, C = [M @ P for P in [B,C]]

    d1,d2,d3,d4 = map(np.linalg.norm, [C-A,B-C,D-B,A-D])
    
    return d1,d2,d3,d4
    
    
def qorder_juan(points):
    A,B,C,D = points
    A,B,C,D = force_AB_maxdist([A,B,C,D])

    Ap,Bp,Cp,Dp = quad_coords_juan(A,B,C,D)

    if not distance(Ap,Bp)<distance(Ap,Cp):
        B, C = C, B

    dL = list(map(np.linalg.norm, [C-A, B-C, D-B, A-B]))
    (A,B,C,D), dL = zip(*sorted(zip([A,B,C,D], dL), key=lambda x:x[1]))

    return A,B,C,D
