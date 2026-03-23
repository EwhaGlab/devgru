import os
import numpy as np
import yaml
import glob
import cv2
import math
import sys

PI = math.pi
def xyzrpy_to_htm( xyzrpy ):
    x = xyzrpy[0]; y = xyzrpy[1]; z = xyzrpy[2]
    rol = xyzrpy[3]; pit = xyzrpy[4]; yaw = xyzrpy[5]
    htm = np.eye(4)
    htm[0:3,0:3] = rpy2rot( rol, pit, yaw  )
    htm[0,3] = x; htm[1,3] = y; htm[2,3] = z
    return htm

def rpy2rot(r, p, y):
    sx = np.sin(r); cx = np.cos(r)
    sy = np.sin(p); cy = np.cos(p)
    sz = np.sin(y); cz = np.cos(y)
    
    R = np.array( [ [cy*cz, sx*sy*cz-cx*sz, cz*sy*cx+sx*sz],
                    [cy*sz, sx*sy*sz+cx*cz, cx*sy*sz-sx*cz],
                    [-sy,   sx*cy,          cx*cy         ] ])
    return R


def htm_to_xyzrpy(htm):
    rol = math.atan2(htm[2,1], htm[2,2])
    yaw = math.atan2(-htm[0,1]*np.cos(rol)+htm[0,2]*np.sin(rol), htm[1,1]*np.cos(rol)-htm[1,2]*np.sin(rol))
    pit = math.atan2(-htm[2,0], htm[0,0]*np.cos(yaw)+ htm[1,0]*np.sin(yaw))
    X= htm[0,3]
    Y= htm[1,3]
    Z= htm[2,3]    
    return X, Y, Z, rol, pit, yaw   
 
def quat_to_htm( q ):
    # q = [w x y z]
    htm      = np.eye(4)
    htm[0,0] = 1 - 2*(q[2]*q[2]+q[3]*q[3])
    htm[0,1] = 2*(q[1]*q[2]-q[0]*q[3])
    htm[0,2] = 2*(q[1]*q[3]+q[0]*q[2])
    
    htm[1,0] = 2*(q[1]*q[2]+q[0]*q[3])
    htm[1,1] = 1 - 2*(q[1]*q[1]+q[3]*q[3])
    htm[1,2] = 2*(q[2]*q[3]-q[0]*q[1])
    
    htm[2,0] = 2*(q[1]*q[3]-q[0]*q[2])
    htm[2,1] = 2*(q[2]*q[3]+q[0]*q[1])
    htm[2,2] = 1 - 2*(q[1]*q[1]+q[2]*q[2])
    htm[3,3] = 1
    return htm

def htm_to_quat( htm ):
    ds = 1 + htm[0,0] + htm[1,1] + htm[2,2]
    dx = 1+htm[0,0]-htm[1,1]-htm[2,2]
    dy = 1-htm[0,0]+htm[1,1]-htm[2,2]
    dz = 1-htm[0,0]-htm[1,1]+htm[2,2]
    quat = np.zeros(4, dtype='float32')

    if( (ds >=dx) and (ds >= dy) and (ds >= dz)):
        quat[0] = (ds**0.5)/2
        quat[1] = (htm[2,1] - htm[1,2])/(4*quat[0])
        quat[2] = (htm[0,2] - htm[2,0])/(4*quat[0])
        quat[3] = (htm[1,0] - htm[0,1])/(4*quat[0])

    elif ((dx >= ds) and (dx >= dy) and (dx >= dz)):
        quat[1] = (dx**0.5)/2
        quat[2] = (htm[1,0] + htm[0,1])/(4*quat[1])
        quat[3] = (htm[2,0] + htm[0,2])/(4*quat[1])
        quat[0] = (htm[2,1] - htm[1,2])/(4*quat[1])

    elif ((dy >= ds) and (dy >= dx) and (dy >= dz)):
        quat[2] = (dy**0.5)/2
        quat[1] = (htm[1,0] + htm[0,1])/(4*quat[2])
        quat[3] = (htm[2,1] + htm[1,2])/(4*quat[2])
        quat[0] = (htm[0,2] - htm[2,0])/(4*quat[2])
	
    elif ((dz >= ds) and (dz >= dx) and (dz >= dy)):
        quat[3] = (dz**0.5)/2
        quat[1] = (htm[2,0] + htm[0,2])/(4*quat[3])
        quat[2] = (htm[2,1] + htm[1,2])/(4*quat[3])
        quat[0] = (htm[1,0] - htm[0,1])/(4*quat[3])
        
    quat_norm = quat / np.linalg.norm(quat)
    return quat_norm

def rpy2quat(r, p, y):
    R = rpy2rot(r, p, y)
    htm = np.eye(4)
    htm[0:3,0:3] = R
    q = htm_to_quat(htm)
    return q

def quat2rpy(*args): # outputs radian
    """
    Convert quaternion(s) [w, x, y, z] -> roll, pitch, yaw in radians.
    Works with single or batch input.
    """
    # ---- parse input ----
    if len(args) == 1:
        q = np.asarray(args[0], dtype=float)
        if q.shape[-1] != 4:
            raise ValueError("Last dimension must be 4 for [w, x, y, z].")
    elif len(args) == 4:
        q = np.asarray(args, dtype=float)
        q = np.moveaxis(q, 0, -1)
    else:
        raise ValueError("Provide either 4 values or a single array of shape (..., 4).")

    # ---- normalize ----
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    q = np.where(norm > 0, q / norm, q)

    w, x, y, z = np.moveaxis(q, -1, 0)
    ysqr = y * y

    # roll
    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x * x + ysqr)
    roll = np.arctan2(t0, t1)

    # pitch
    t2 = 2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1.0, 1.0)
    pitch = np.arcsin(t2)

    # yaw
    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (ysqr + z * z)
    yaw = np.arctan2(t3, t4)

    return roll, pitch, yaw # radian



def Left2Right( H_l ):
    rx = H_l[0,0]; lx = H_l[0,1]; ux = H_l[0,2]; px = H_l[0,3]
    rz = H_l[1,0]; lz = H_l[1,1]; uz = H_l[1,2]; pz = H_l[1,3]
    ry = H_l[2,0]; ly = H_l[2,1]; uy = H_l[2,2]; py = H_l[2,3]
    H_r = np.array( [[rx, ux, lx, px], [ry, uy, ly, py], [rz, uz, lz, pz], [0, 0, 0, 1]] )  
    
    return H_r

#def Right2Left( R_r ):
    #rx = R_r[0,0]; ry = R_r[0,1]; rz = R_r[0,2];
    #ux = R_r[1,0]; uy = R_r[1,1]; uz = R_r[1,2];
    #lx = R_r[2,0]; ly = R_r[2,1]; lz = R_r[2,2];
    #px = R_r[3,0]; py = R_r[3,1]; pz = R_r[3,2];
    
    #R_l = np.array( [[rx, rz, ry], [lx, lz, ly], [ux, uz, uy], 
    
def quat_dist(q1, q2):
    if ( len(q1.shape) == 1):
        q1 = q1[None,...]
    if ( len(q2.shape) == 1):
        q2 = q2[None,...]
    
    q2_conj = -q2.copy()
    q2_conj[:,0] = q2.copy()[:,0]
    
    d1 = np.linalg.norm(q1 - q2)
    d2 = np.linalg.norm(q1 - q2_conj)
    return min(d1, d2)

def quat_ang_dist(q1, q2):
    "Computes angular distance between two unit quaternions in radians. q1, q2: [4] or [N, 4] arrays, format [qw, qx, qy, qz] or [qw, qz] for 2D"

    q1 = np.asarray(q1, dtype=np.float64)
    q2 = np.asarray(q2, dtype=np.float64)

    q1 = np.atleast_2d(q1)
    q2 = np.atleast_2d(q2)

    # same format per call (both 2 or both 4)
    D1, D2 = q1.shape[-1], q2.shape[-1]
    assert D1 == D2 and D1 in (2, 4), f"Expected last dim 2 or 4 (and equal), got {D1} vs {D2}"

    # L2-normalize
    q1 = q1 / (np.linalg.norm(q1, axis=-1, keepdims=True) + 1e-12)
    q2 = q2 / (np.linalg.norm(q2, axis=-1, keepdims=True) + 1e-12)

    if D1 == 4:
        # [qw,qx,qy,qz]
        dot = np.sum(q1 * q2, axis=-1)
    else:
        # [qw,qz]
        dot = q1[:, 0] * q2[:, 0] + q1[:, 1] * q2[:, 1]

    dot = np.clip(dot, -1.0, 1.0)
    cos_half = np.abs(dot)
    angle = 2 * np.arccos( np.clip( cos_half, 0.0, 1.0 ) )  # [0, π]
    sin2_half = 1.0 - cos_half * cos_half                   # [0, 1]

    return sin2_half.squeeze(), angle.squeeze()  # shape: [] or [N]

def theta_err_btwn_quats( q1, q2, signed=True ):

    q1 = np.asarray(q1, dtype=np.float64)
    q2 = np.asarray(q2, dtype=np.float64)
    #q1 = np.atleast_2d(q1); q2 = np.atleast_2d(q2) # (B, 4)
    q1 = q1 / (np.linalg.norm(q1, axis=-1, keepdims=True) + 1e-12)
    q2 = q2 / (np.linalg.norm(q2, axis=-1, keepdims=True) + 1e-12)
    D = q1.shape[-1]

    if D == 2:
        #print(q1.shape)   (B, 2)
        qw1, qz1 = q1 # [qw]
        qw2, qz2 = q2
        q1 = np.array([qw1, 0, 0, qz1], dtype=np.float64)
        q2 = np.array([qw2, 0, 0, qz2], dtype=np.float64)

    q_dist, ang_dist = quat_ang_dist(q1, q2)

    return ang_dist # radian

############################################3
# 3D to 2D pinhole projection
def pinhole_projection( K, xyz_c ) -> np.ndarray: # xyz_c to uv : 3xN to 2xN mapping
    uvS = np.matmul(K, xyz_c)

    uv = uvS[0:2].copy()
    uv[0] = uvS[0] / uvS[2]
    uv[1] = uvS[1] / uvS[2]
    return uv


def slerp( q1, q2, t ) -> np.ndarray:
#   SLERP quaternion slerp
#   computes the slerp of value t between quaternions q1 and q2
    q1 = np.asarray(q1)
    q2 = np.asarray(q2)
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)

    one_eps = 1.0 - sys.float_info.epsilon 
    cos_omega = np.round( np.dot(q1, q2), 3)
    assert( abs(cos_omega) <= 1.0 )
    #print("cos_omega", cos_omega)
    omega = math.acos(cos_omega)

    if( abs(cos_omega) >= one_eps): # near 0 ang case
        S1 = 1 - t
        S2 = t
    else:
        # theta is the angle between the 2 quaternions
        sin_omega = math.sin(omega)
        S1 = math.sin( ( 1.0 - t ) * omega) / sin_omega
        S2 = math.sin( ( t * omega) ) / sin_omega

    if(cos_omega < 0):
        S1 = -S1
    
    q3 = S1 * q1 + S2 * q2
    q3 = q3 / np.linalg.norm(q3)
    return q3

def normalizeAngle( fangle_rad, type: bool= 1 ) -> float:

    assert( fangle_rad >= -2*PI and fangle_rad <= 2*PI )
    ang_deg = fangle_rad * 180 / PI

    if type == 0: # normalize btwn 0 ~ 360
        ang_deg_norm = ang_deg % 360
        if(ang_deg < 0):
            ang_deg += 360
        return ang_deg * PI / 180
    else: # keep ang btwn -179 ~ 180
        ang_deg = (ang_deg + 180) % 360
        if(ang_deg < 0):
            ang_deg += 360
        return (ang_deg - 180) / 180 * PI

    #     while (ang_deg <= 180):
    #         ang_deg +=360
    #
    #     while (ang_deg > 180):
    #         ang_deg -= 180
    #     return ang_deg * PI / 180

