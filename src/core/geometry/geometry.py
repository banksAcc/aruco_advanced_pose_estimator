# pc/app/algo/geometry.py
import numpy as np
import cv2 as cv
from typing import List, Optional, Tuple

def quat_from_R(R: np.ndarray) -> np.ndarray:
    """Converte matrice di rotazione 3x3 in Quaternione (w, x, y, z)."""
    tr = np.trace(R)
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        w = 0.25 * S
        x = (R[2,1] - R[1,2]) / S
        y = (R[0,2] - R[2,0]) / S
        z = (R[1,0] - R[0,1]) / S
    else:
        idx = np.argmax([R[0,0], R[1,1], R[2,2]])
        if idx == 0:
            S = np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2]) * 2
            w = (R[2,1] - R[1,2]) / S
            x = 0.25 * S
            y = (R[0,1] + R[1,0]) / S
            z = (R[0,2] + R[2,0]) / S
        elif idx == 1:
            S = np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2]) * 2
            w = (R[0,2] - R[2,0]) / S
            x = (R[0,1] + R[1,0]) / S
            y = 0.25 * S
            z = (R[1,2] + R[2,1]) / S
        else:
            S = np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1]) * 2
            w = (R[1,0] - R[0,1]) / S
            x = (R[0,2] + R[2,0]) / S
            y = (R[1,2] + R[2,1]) / S
            z = 0.25 * S     
    return np.array([w, x, y, z])

def _compute_mean_pose(poses: List[np.ndarray], weights: np.ndarray) -> np.ndarray:
    """Helper interno per calcolare la media pesata."""
    if not poses:
        return np.eye(4)

    w_sum = np.sum(weights)
    # Evitiamo divisioni per zero
    norm_weights = (weights / w_sum) if w_sum > 1e-9 else (np.ones(len(weights))/len(weights))

    # Traslazione
    translations = np.array([P[:3, 3] for P in poses])
    t_mean = np.average(translations, axis=0, weights=norm_weights)
    
    # Rotazione (Media matrici + SVD)
    R_sum = np.zeros((3, 3))
    for i, P in enumerate(poses):
        R_sum += P[:3, :3] * norm_weights[i]
        
    U, _, Vt = np.linalg.svd(R_sum)
    R_mean = U @ Vt
    
    # Fix riflessione SVD
    if np.linalg.det(R_mean) < 0:
        U[:, -1] *= -1
        R_mean = U @ Vt
        
    T_out = np.eye(4)
    T_out[:3, :3] = R_mean
    T_out[:3, 3] = t_mean
    return T_out

from typing import List, Optional, Tuple
import numpy as np

def average_poses(
    poses_4x4: List[np.ndarray],
    marker_rotations: List[np.ndarray],    
    weights: Optional[List[float]] = None,
    weight_exponent: float = 2.0,
    outlier_distance_threshold: Optional[float] = None,
    max_angle_deg: float = 80  
) -> Tuple[Optional[np.ndarray], List[int], List[int]]:
    """
    Calcola la media pesata delle pose rimuovendo marker troppo inclinati e outlier.
    
    Returns:
        - T_final: La posa media 4x4 (o None se fallisce)
        - inlier_indices: Indici dei marker usati nel calcolo finale
        - tilted_indices: Indici dei marker scartati perché oltre max_angle_deg
    """
    n = len(poses_4x4)
    if n == 0:
        raise ValueError("Lista pose vuota.")
    
    min_cos_theta = np.cos(np.radians(max_angle_deg))

    valid_poses = []
    valid_raw_weights = []
    initial_indices = []
    tilted_indices = []

    # --- 1. FILTRO A MONTE: Angolo di incidenza ---
    for i in range(n):
        R_marker = marker_rotations[i]
        # Prodotto scalare tra asse Z marker e asse Z camera
        cos_theta = abs(R_marker[2, 2])
        
        if cos_theta >= min_cos_theta:
            valid_poses.append(poses_4x4[i])
            valid_raw_weights.append(weights[i] if weights is not None else 1.0)
            initial_indices.append(i)
        else:
            angle_actual = np.degrees(np.arccos(min(1.0, cos_theta)))
            print(f"  > Scartato marker {i}: troppo inclinato ({angle_actual:.1f}°)")
            tilted_indices.append(i) # Salviamo l'indice del marker inclinato

    # Se nessun marker supera il filtro inclinazione
    if not valid_poses:
        print(f" [AVG] Fallimento: Tutti i {n} marker sono troppo inclinati.")
        return None, [], tilted_indices
    
    proc_weights = np.array(valid_raw_weights, dtype=float)
    proc_weights = np.maximum(proc_weights, 1e-3)
    proc_weights = np.power(proc_weights, weight_exponent)

    # 2. Calcolo della media preliminare
    T_initial = _compute_mean_pose(valid_poses, proc_weights)
    
    # Se non c'è soglia di outlier, restituiamo i risultati del primo filtro
    if outlier_distance_threshold is None or outlier_distance_threshold <= 0:
        return T_initial, initial_indices, tilted_indices

    # 3. Filtraggio Outlier (Distanza dalla media preliminare)
    t_center = T_initial[:3, 3]
    
    final_poses = []
    final_weights = []
    inlier_indices = []

    for i, pose in enumerate(valid_poses):
        w = proc_weights[i]
        dist = np.linalg.norm(pose[:3, 3] - t_center)
        
        if dist <= outlier_distance_threshold:
            final_poses.append(pose)
            final_weights.append(w)
            inlier_indices.append(initial_indices[i])
        else:
            print(f"  > Dropped marker {initial_indices[i]} (Dist: {dist:.3f})")

    # 4. CONTROLLO FINALE
    if not final_poses:
        print(f" [AVG] Fallimento: Nessun marker entro la soglia di distanza ({outlier_distance_threshold}m).")
        return None, [], tilted_indices

    # Ricalcoliamo la media finale
    T_final = _compute_mean_pose(final_poses, np.array(final_weights))
    
    return T_final, inlier_indices, tilted_indices
