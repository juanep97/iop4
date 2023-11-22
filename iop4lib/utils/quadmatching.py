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

def hash_ish_old(points):
    P1,P2,P3,P4 = points     
    P1,P2,P3,P4 = order(points)
    d1,d2,d3,d4 = map(np.linalg.norm, [P2-P1,P3-P2,P4-P3,P1-P4])
    return d1,d2,d3,d4



def quad_coords_ish(A,B,C,D):

    (A,B,C,D), idx = force_AB_maxdist([A,B,C,D])

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

    FX = np.array([[-1,0],[0,1]])
    FY = np.array([[1,0],[0,-1]])

    # begin track the idx

    if C[0] + D[0] > 0:
        idx[2], idx[3] = idx[3], idx[2]
        idx[0], idx[1] = idx[1], idx[0]
    if C[0] > D[0]:
        idx[2], idx[3] = idx[3], idx[2]
    
    # end track the idx

    if C[0] + D[0] > 0:
        C,D = [FX@P for P in [C,D]]

    if C[1] + D[1] > 0:
        C,D = [FY@P for P in [C,D]]

    if C[0] > D[0]:
        C,D = D,C

    return (A,B,C,D), idx


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
    
    result_idx = [i,j]
    result = [points[i], points[j]]

    result_idx.extend([k for k in range(len(points)) if k not in [i,j]])
    result.extend([points[k] for k in range(len(points)) if k not in [i,j]])
    
    return result, result_idx
    

def hash_ish(points):
    
    A,B,C,D = points
    (A,B,C,D), idx = quad_coords_ish(A,B,C,D)
    d1,d2,d3,d4 = map(np.linalg.norm, [C-A,D-C,B-D,A-B])

    return d1,d2,d3,d4
    
    
def qorder_ish(points):

    (Ap,Bp,Cp,Dp), idx_quad_ord = quad_coords_ish(points[0],points[1],points[2],points[3])

    pts_quad_ord = [points[i] for i in idx_quad_ord]

    # cx = np.mean([p[0] for p in [Ap,Bp,Cp,Dp]])
    # cy = np.mean([p[1] for p in [Ap,Bp,Cp,Dp]])
    # idx_sorting, _ = list(zip(*sorted(enumerate([Ap,Bp,Cp,Dp]), key=lambda p: np.arctan2(p[1][1]-cy, p[1][0]-cx))))
    # A,B,C,D = [pts_quad_ord[i] for i in idx_sorting]

    A,B,C,D = pts_quad_ord

    return A,B,C,D


def find_linear_transformation(P1, P2):
    P1, P2 = np.array(P1), np.array(P2)
    
    # Center points
    P1_mean, P2_mean = np.mean(P1, axis=0), np.mean(P2, axis=0)
    P1_centered, P2_centered = P1 - P1_mean, P2 - P2_mean
    
    # SVD
    H = P1_centered.T @ P2_centered
    U, _, Vt = np.linalg.svd(H)
    
    # Rotation + Scaling Matrix
    R = Vt.T @ U.T
    
    # Translation Vector
    t = P2_mean - R @ P1_mean
    
    return R, t