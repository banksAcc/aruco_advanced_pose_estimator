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

def average_poses(
    poses_4x4: List[np.ndarray], 
    weights: Optional[List[float]] = None,
    weight_exponent: float = 2.0,
    outlier_distance_threshold: Optional[float] = None
) -> Tuple[np.ndarray, List[int]]:
    """
    Calcola la media pesata delle pose rimuovendo gli outlier.
    Restituisce (Posa_Media, Indici_Inlier).
    """
    n = len(poses_4x4)
    if n == 0:
        raise ValueError("Lista pose vuota.")

    # 1. Preparazione Pesi
    if weights is None:
        raw_weights = np.ones(n)
    else:
        raw_weights = np.array(weights, dtype=float)
    
    # Evitiamo pesi zero e applichiamo l'esponente
    raw_weights = np.maximum(raw_weights, 1e-3)
    proc_weights = np.power(raw_weights, weight_exponent)

    # 2. Calcolo della media preliminare (su TUTTI i marker)
    # Assumo che _compute_mean_pose sia definita altrove nel tuo codice
    T_initial = _compute_mean_pose(poses_4x4, proc_weights)
    
    # Se non c'è soglia di outlier, abbiamo finito qui
    if outlier_distance_threshold is None or outlier_distance_threshold <= 0:
        return T_initial

    # 3. Filtraggio Outlier (Distanza dalla media preliminare)
    t_center = T_initial[:3, 3]
    
    final_poses = []
    final_weights = []
    inlier_indices = []

    for i in range(n):
        pose = poses_4x4[i]
        w = proc_weights[i]
        
        # Calcola distanza euclidea dal centro calcolato
        dist = np.linalg.norm(pose[:3, 3] - t_center)
        
        if dist <= outlier_distance_threshold:
            final_poses.append(pose)
            final_weights.append(w)
            inlier_indices.append(i) # <--- Segna come inlier
        else:
            print(f"  > Dropped marker {i} (Dist: {dist:.3f})")

    # 4. Calcolo finale sui soli inlier
    if not final_poses:
        # Se abbiamo scartato tutto (troppo restrittivi), torniamo la media iniziale come fallback
        print("  [AVG] WARNING: All markers rejected. Returning initial mean.")
        return T_initial, list(range(n)) # Tutti inlier come fallback

    # Ricalcoliamo la media solo con i marker "buoni"
    T_final = _compute_mean_pose(final_poses, np.array(final_weights))
    
    return T_final, inlier_indices