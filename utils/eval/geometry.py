import numpy as np
from transforms3d.quaternions import quat2mat, mat2quat


# The skew-symmetric matrix of vector
skew = lambda v: np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])

# Essential matrix & fundamental matrix 
ess2fund = lambda K1, K2, E: np.linalg.inv(K2).T @ E @ np.linalg.inv(K1)
ess2fund_inv = lambda K1_inv, K2_inv, E: K2_inv.T @ E @ K1_inv
fund2ess = lambda F, K2, K1: K2.T @ F @ K1

# Camera relative pose to fundamental matrix
pose2ess = lambda R, t: skew(t.reshape(3,)) @ R
pose2fund = lambda K1, K2, R, t: np.linalg.inv(K2).T @ R @ K1.T @ skew((K1 @ R.T).dot(t.reshape(3,)))
pose2fund_inv = lambda K1, K2_inv, R, t: K2_inv.T @ R @ K1.T @ skew((K1 @ R.T).dot(t))

# Normalize fundamental matrix
normF = lambda F: F / F[-1,-1] # Normalize F by the last value
normalize = lambda A:  A / np.linalg.norm(A)

def compose_projection_matrix(R, t):
    """Construct projection matrix 
    Args:
        - R: rotation matrix, size (3,3);
        - t: translation vector, size (3,);
    Return:
        - projection matrix [R|t], size (3,4)
    """
    return np.hstack([R, np.expand_dims(t, axis=1)])

def matches2relapose_cv(p1, p2, K1, K2, rthres=1):
    import cv2
    # Move back to image center based coordinates
    f1, f2,  = K1[0,0], K2[0, 0]   
    pc1 = np.array([K1[:2, 2]])
    pc2 = np.array([K2[:2, 2]])

    # Rescale to im2 's focal setting
    p1 = (p1 - pc1) * f2 / f1
    p2 = (p2 - pc2) 
    K = np.array([[f2, 0, 0],
                 [0, f2, 0],
                 [0, 0, 1]])      
    E, inls = cv2.findEssentialMat(p1, p2, cameraMatrix=K, method=cv2.FM_RANSAC, threshold=rthres)            
    inls = np.where(inls > 0)[0]
    _, R, t, _ = cv2.recoverPose(E, p1[inls], p2[inls], K)    
    return E, inls, R, t

def matches2relapose_degensac(p1, p2, K1, K2, rthres=1):
    import pydegensac
    import cv2    
    
    # Move back to image center based coordinates
    f1, f2  = K1[0,0], K2[0, 0]   
    pc1 = np.array([K1[:2, 2]])
    pc2 = np.array([K2[:2, 2]])

    # Rescale to im2 's focal setting
    p1 = (p1 - pc1) * f2 / f1
    p2 = (p2 - pc2) 
    K = np.array([[f2, 0, 0],
                 [0, f2, 0],
                 [0, 0, 1]])
    K1 = K2 = K
    
    F, inls = pydegensac.findFundamentalMatrix(p1, p2, rthres)    
    E = fund2ess(F, K1, K2)
    inls = np.where(inls > 0)[0]    
    _, R, t, _ = cv2.recoverPose(E, p1[inls], p2[inls], K)
    return E, inls, R, t

def abs2relapose(c1, c2, q1, q2):
    """Calculate relative pose between two cameras
    Args:
    - c1: absolute position of the first camera
    - c2: absolute position of the second camera
    - q1: orientation quaternion of the first camera
    - q2: orientation quaternion of the second camera
    Return:
    - (t12, q12): relative pose giving the transformation from the 1st camera to the 2nd camera coordinates, 
                  t12 is translation, q12 is relative rotation quaternion 
    """
    r1 = quat2mat(q1)
    r2 = quat2mat(q2)
    r12 = r2.dot(r1.T)
    q12 = mat2quat(r12)
    t12 = r2.dot(c1 - c2)
    return (t12, q12)
          