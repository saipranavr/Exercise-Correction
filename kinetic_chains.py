import numpy as np

# Based on OpenPose 25-joint model, which is standard for NTU-RGB+D
ntu_joints = {
    "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
    "LShoulder": 5, "LElbow": 6, "LWrist": 7, "MidHip": 8, "RHip": 9,
    "RKnee": 10, "RAnkle": 11, "LHip": 12, "LKnee": 13, "LAnkle": 14,
    "REye": 15, "LEye": 16, "REar": 17, "LEar": 18, "LBigToe": 19,
    "LSmallToe": 20, "LHeel": 21, "RBigToe": 22, "RSmallToe": 23, "RHeel": 24
}

def get_kinetic_chain_matrix(num_points=25):
    A = np.zeros((4, num_points, num_points))

    # 1. Intrinsic Core (stabilizers along the spine)
    # Connections: Hips -> Neck
    intrinsic_core = [
        (ntu_joints["MidHip"], ntu_joints["RHip"]), (ntu_joints["MidHip"], ntu_joints["LHip"]),
        (ntu_joints["MidHip"], ntu_joints["Neck"]),
        (ntu_joints["RHip"], ntu_joints["RShoulder"]), # Less direct, but part of core stabilization
        (ntu_joints["LHip"], ntu_joints["LShoulder"])
    ]
    for i, j in intrinsic_core:
        A[0, i, j] = 1
        A[0, j, i] = 1

    # 2. Deep Longitudinal System (DLS) - Force transmission up the body
    # Connections: Foot -> Knee -> Hip -> Spine -> Shoulder
    dls_right = [
        (ntu_joints["RAnkle"], ntu_joints["RKnee"]), (ntu_joints["RKnee"], ntu_joints["RHip"]),
        (ntu_joints["RHip"], ntu_joints["Neck"])
    ]
    dls_left = [
        (ntu_joints["LAnkle"], ntu_joints["LKnee"]), (ntu_joints["LKnee"], ntu_joints["LHip"]),
        (ntu_joints["LHip"], ntu_joints["Neck"])
    ]
    for i, j in dls_right + dls_left:
        A[1, i, j] = 1
        A[1, j, i] = 1

    # 3. Lateral Stabilizing (LS) - Side-to-side stability
    # Connections: Hip -> Opposite side QL/Oblique -> Opposite Shoulder
    # Simplified as: Ankle -> Knee -> Hip -> Opposite Shoulder
    ls_chain = [
        (ntu_joints["RAnkle"], ntu_joints["RHip"]), (ntu_joints["RHip"], ntu_joints["LShoulder"]),
        (ntu_joints["LAnkle"], ntu_joints["LHip"]), (ntu_joints["LHip"], ntu_joints["RShoulder"])
    ]
    for i, j in ls_chain:
        A[2, i, j] = 1
        A[2, j, i] = 1

    # 4. Anterior Oblique System (AOS) - Rotational force
    # Connections: Hip -> Adductor -> Oblique -> Opposite Shoulder/Chest
    # Simplified as: Hip -> Opposite Shoulder
    aos_chain = [
        (ntu_joints["RHip"], ntu_joints["LShoulder"]),
        (ntu_joints["LHip"], ntu_joints["RShoulder"])
    ]
    for i, j in aos_chain:
        A[3, i, j] = 1
        A[3, j, i] = 1
        
    return A
