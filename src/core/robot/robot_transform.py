"""
Gestisce le trasformazioni di coordinate tra Camera e Base Robot.
"""
from typing import Optional
import numpy as np
import cv2

def get_base_to_camera_matrix(
    x_mm: float,
    y_mm: float,
    z_mm: float,
    phi_deg: float,
    theta_deg: float,
    psi_deg: float
) -> np.ndarray:
    """
    Calcola la matrice estrinseca T_base_cam (4x4) usando i parametri forniti
    e la convenzione di Eulero ZYZ (Phi, Theta, Psi).

    Se chiamata senza argomenti, usa i valori di default definiti nel modulo.

    Args:
        x_mm, y_mm, z_mm: Traslazione in millimetri.
        phi_deg:   1° rotazione attorno a Z (gradi).
        theta_deg: 2° rotazione attorno a Y (gradi).
        psi_deg:   3° rotazione attorno a Z (gradi).

    Returns:
        np.ndarray: Matrice 4x4 T_base_cam.
    """
# 1. Conversione mm -> metri
    x = x_mm / 1000.0
    y = y_mm / 1000.0
    z = z_mm / 1000.0
    
    # 2. Conversione Gradi -> Radianti
    phi   = np.radians(phi_deg)
    theta = np.radians(theta_deg)
    psi   = np.radians(psi_deg)
    
    # Precalcolo seni e coseni
    c_phi, s_phi = np.cos(phi), np.sin(phi)
    c_th,  s_th  = np.cos(theta), np.sin(theta)
    c_psi, s_psi = np.cos(psi), np.sin(psi)
    
    # Costruzione Matrici di Rotazione (Eulero ZYZ)
    # R = Rz(phi) * Ry(theta) * Rz(psi)
    
    Rz_phi = np.array([
        [c_phi, -s_phi, 0], 
        [s_phi,  c_phi, 0], 
        [0,      0,     1]
    ])
    
    Ry_theta = np.array([
        [c_th,  0, s_th], 
        [0,     1, 0   ], 
        [-s_th, 0, c_th]
    ])
    
    Rz_psi = np.array([
        [c_psi, -s_psi, 0], 
        [s_psi,  c_psi, 0], 
        [0,      0,     1]
    ])
    
    # Rotazione finale
    R = Rz_phi @ Ry_theta @ Rz_psi
    
    # Costruzione Matrice Omogenea 4x4
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [x, y, z]
    
    return T

def transform_camera_to_robot(
    T_cam_obj: np.ndarray, 
    T_base_cam: Optional[np.ndarray] = None
) -> Optional[np.ndarray]:
    """
    Converte una posa completa (Matrice 4x4) dal frame Camera al frame Robot Base.
    
    Args:
        T_cam_obj: Posa dell'oggetto in coordinate Camera (4x4).
        T_base_cam: Matrice di trasformazione Base -> Camera (4x4).
                    Se None, la trasformazione non può essere calcolata.

    Returns:
        T_base_obj (4x4) se T_base_cam è valido, altrimenti None.
    """
    if T_base_cam is None:
        return None

    # Moltiplicazione matriciale diretta: Base = (Base->Cam) * (Cam->Obj)
    T_base_obj = T_base_cam @ T_cam_obj
    
    return T_base_obj