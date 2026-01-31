"""
Gestisce le trasformazioni di coordinate tra Camera e Base Robot.
"""
import numpy as np
import cv2

# Configurazione Hardcoded (o caricabile da config in futuro)
# Unità: mm e gradi
EXT_X_MM = 871.432
EXT_Y_MM = -61.7029
EXT_Z_MM = 760.249
EXT_A_DEG = 149.045   # Z (Phi)
EXT_B_DEG = 150.698   # Y (Theta)
EXT_C_DEG = -90.0639  # Z (Psi)

def get_base_to_camera_matrix() -> np.ndarray:
    """Calcola la matrice estrinseca T_base_cam (4x4) usando Eulero ZYZ."""
    # 1. mm -> metri
    x = EXT_X_MM / 1000.0
    y = EXT_Y_MM / 1000.0
    z = EXT_Z_MM / 1000.0
    
    # 2. Gradi -> Radianti
    a = np.radians(EXT_A_DEG)
    b = np.radians(EXT_B_DEG)
    c = np.radians(EXT_C_DEG)
    
    c_a, s_a = np.cos(a), np.sin(a)
    c_b, s_b = np.cos(b), np.sin(b)
    c_c, s_c = np.cos(c), np.sin(c)
    
    # R = Rz(a) * Ry(b) * Rz(c)
    Rz_a = np.array([[c_a, -s_a, 0], [s_a, c_a, 0], [0, 0, 1]])
    Ry_b = np.array([[c_b, 0, s_b], [0, 1, 0], [-s_b, 0, c_b]])
    Rz_c = np.array([[c_c, -s_c, 0], [s_c, c_c, 0], [0, 0, 1]])
    
    R = Rz_a @ Ry_b @ Rz_c
    
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [x, y, z]
    return T

def transform_camera_to_robot(rvec_cam: np.ndarray, tvec_cam: np.ndarray, T_base_cam: np.ndarray):
    """
    Converte una posa dal frame Camera al frame Robot Base.
    Restituisce (rvec_robot, tvec_robot).
    """
    # Converti rvec/tvec in matrice 4x4
    R_cam, _ = cv2.Rodrigues(rvec_cam)
    T_cam_obj = np.eye(4)
    T_cam_obj[:3, :3] = R_cam
    T_cam_obj[:3, 3] = tvec_cam.flatten()
    
    # T_base_obj = T_base_cam * T_cam_obj
    T_base_obj = T_base_cam @ T_cam_obj
    
    # Estrai rvec/tvec
    rvec_base, _ = cv2.Rodrigues(T_base_obj[:3, :3])
    tvec_base = T_base_obj[:3, 3]
    
    return rvec_base.flatten(), tvec_base.flatten()
