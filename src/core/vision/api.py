# pc/app/algo/api.py
from typing import Dict, Any, Optional, Mapping, List, Tuple
import numpy as np
import cv2 as cv

from .detect import detect_markers
from .pnp import estimate_marker_poses
from ..geometry.geometry import average_poses, quat_from_R
from .viz import draw_sphere_overlay, draw_detected_markers_with_projection, draw_large_axes

# --- CONFIGURAZIONE DEFAULT (Da spostare in JSON/YAML) ---
# Parametri geometrici utensile
GEO_RADIUS = 0.05818      # Raggio sfera (metri)
Z_SHIFT_TIP = 0.20620     # Distanza Centro Sfera -> Punta (metri)

# Parametri algoritmi
MIN_MARKER_AREA = 100.0   # Area minima in pixel
SUBPIX_ITER = 500         # Iterazioni subpixel
SUBPIX_EPS = 0.0001       # Epsilon subpixel
WEIGHT_EXPONENT = 1.5     # Esponente per pesatura area
OUTLIERS_THRESHOLD = 0.030

DEFAULT_MARKER_MAP = {
    1: "H5", 2: "H2", 3: "P0", 4: "H3", 5: "H0", 6: "H6", 7: "P2", 8: "P8", 
    10: "H8", 9: "H4", 11: "P1", 12: "H9", 13: "P7", 14: "P6", 15: "H15", 
    16: "P4", 17: "H1", 18: "H7", 19: "H14", 20: "H17", 21: "H19", 22: "P3", 
    23: "H13", 24: "P11", 25: "P5", 26: "P10", 27: "H10", 28: "H16", 
    29: "H11", 30: "H18", 31: "H12",
}

def transform_to_robot_base(T_cam_obj: np.ndarray, T_base_cam: Optional[np.ndarray]) -> np.ndarray:
    """
    Applica la trasformazione estrinseca Camera -> Base Robot.
    Se T_base_cam è None, restituisce la posa originale (Base = Camera).
    """
    if T_base_cam is None:
        return T_cam_obj
    return T_base_cam @ T_cam_obj

def estimate_truncated_ico_from_image(
    image: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    transforms: Mapping[str, np.ndarray], # Map: FaceID -> T_face_body
    aruco_dict: str,
    marker_size: float,
    T_base_cam: Optional[np.ndarray] = None, # Nuova: Estrinseca Robot
    return_overlay: bool = False,
    marker_map: Dict[int, str] = DEFAULT_MARKER_MAP,
) -> Dict[str, Any]:

    # =========================================================================
    # 1. DETECTION & SUBPIXEL REFINEMENT
    # =========================================================================
    detections = detect_markers(image, aruco_dict)
    
    if detections:
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, SUBPIX_ITER, SUBPIX_EPS)
        for det in detections:
            cv.cornerSubPix(gray, det.corners, (5, 5), (-1, -1), criteria)

    # =========================================================================
    # 2. FILTRO AREA
    # =========================================================================
    valid_detections = [d for d in detections if d.area_px >= MIN_MARKER_AREA]
    
    if not valid_detections:
        raise ValueError("Nessun marker valido trovato (Filtro Area).")

    # =========================================================================
    # 3. STIMA POSA MARKER (PnP)
    # =========================================================================
    # poses[i] corrisponde a valid_detections[i]
    poses = estimate_marker_poses(valid_detections, K, dist, marker_size)
    if not poses:
        raise ValueError("PnP fallito su tutti i marker.")

    # =========================================================================
    # 4. COSTRUZIONE STRUTTURA CANDIDATI (CAM FRAME)
    # =========================================================================
    # Qui calcoliamo dove ogni marker "pensa" che sia il centro della sfera.
    # Non facciamo ancora calcoli visivi complessi, solo matematica pura.
    
    sphere_candidates_cam = []  # Lista di matrici 4x4 (Pose del Centro Sfera)
    weights = []                # Pesi associati (Area)
    
    # Struttura dati per debugging/viz (contiene info per ogni singolo marker valido)
    markers_debug_info = []

    for det, m_pose in zip(valid_detections, poses):
        face_id = marker_map.get(det.id)
        
        # Saltiamo marker non mappati o senza trasformazione geometrica nota
        if not face_id or face_id not in transforms:
            continue

        T_face_body = transforms[face_id]
        # --- DEBUG PRINT ---
            # Stampa la traslazione (x, y, z) per verificare l'unità di misura
            # Se vedi valori tipo [0.058, 0.0, 0.0] -> METRI (Corretto)
            # Se vedi valori tipo [58.18, 0.0, 0.0] -> MILLIMETRI (Errore, devi dividere per 1000 nel JSON o qui)
        t_vec_debug = T_face_body[:3, 3]
        print(f"[DEBUG] Face {face_id} trans: {t_vec_debug} | Norm: {np.linalg.norm(t_vec_debug):.4f}")
        # -------------------
        
        # T_cam_marker: dal Marker alla Camera
        T_cam_marker = np.eye(4)
        T_cam_marker[:3, :3] = m_pose.R
        T_cam_marker[:3, 3] = m_pose.tvec.flatten()
        
        # T_cam_body_candidate: Dove questo marker dice che è il centro sfera
        # (Camera -> Marker) * (Marker -> BodyCenter)
        T_cam_sphere_candidate = T_cam_marker @ T_face_body
        
        sphere_candidates_cam.append(T_cam_sphere_candidate)
        weights.append(det.area_px)
        
        # Salviamo i dati raw per uso futuro (overlay o debug)
        markers_debug_info.append({
            "id": det.id,
            "face": face_id,
            "T_cam_marker": T_cam_marker,         # Posa reale del marker
            "T_cam_candidate": T_cam_sphere_candidate, # Ipotesi centro sfera
            "detection": det
        })

    if not sphere_candidates_cam:
        raise ValueError("Nessun marker corrisponde a facce note nel JSON.")

    # =========================================================================
    # 5. MEDIA PESATA E STABILIZZATA (CENTRO SFERA)
    # =========================================================================
    # average_poses rimuove outlier e calcola la media robusta
    # Output: T_cam_sphere_avg (Posa media del centro sfera rispetto alla camera)
    T_cam_sphere_avg, inlier_indices= average_poses(
        sphere_candidates_cam, 
        weights=weights,
        weight_exponent=WEIGHT_EXPONENT,
        outlier_distance_threshold=OUTLIERS_THRESHOLD # Esempio: scarta se > 2cm dalla media
    )

    # =========================================================================
    # 6. CALCOLO PUNTA PENNA (CAM FRAME)
    # =========================================================================
    # Dalla posa media della sfera, trasliamo lungo Z negativo locale per trovare la punta
    
    R_avg = T_cam_sphere_avg[:3, :3]
    t_sphere_avg = T_cam_sphere_avg[:3, 3]
    
    local_z_axis = R_avg[:, 2]
    
    # Calcolo posizione punta: Centro - (AsseZ * Distanza)
    # Nota: Il segno dipende da come è definito il frame 'Body' nel JSON.
    # Assumiamo che la punta sia lungo -Z rispetto al centro, come nel codice originale.
    t_tip_cam = t_sphere_avg - (local_z_axis * Z_SHIFT_TIP)
    
    # Costruiamo la matrice completa per la punta (stessa rotazione del corpo)
    T_cam_tip = np.eye(4)
    T_cam_tip[:3, :3] = R_avg
    T_cam_tip[:3, 3] = t_tip_cam

    # =========================================================================
    # 7. TRASFORMAZIONE BASE ROBOT (CAM -> BASE)
    # =========================================================================
    # Convertiamo i risultati finali nel sistema di riferimento del robot
    
    T_base_tip = transform_to_robot_base(T_cam_tip, T_base_cam)
    
    # Estraiamo i componenti finali per l'output (Base Frame)
    t_final_base = T_base_tip[:3, 3]
    R_final_base = T_base_tip[:3, :3]
    quat_final_base = quat_from_R(R_final_base)
    rvec_final_base, _ = cv.Rodrigues(R_final_base)

    # =========================================================================
    # 8. DISEGNO OVERLAY (VISUALIZZAZIONE)
    # =========================================================================
    
    overlay_img = None
    if return_overlay:
        
        # Prepariamo le liste per la funzione di disegno
        ghost_projections = []
        is_inlier_list = []

        # Usiamo enumerate per avere l'indice 'i' corretto da confrontare con inlier_indices
        for i, info in enumerate(markers_debug_info):
            # Estraiamo il tvec del candidato centro sfera (Ghost)
            ghost_projections.append(info["T_cam_candidate"][:3, 3])
            
            # Verifichiamo se questo marker (indice i) è un inlier
            is_inlier_list.append(i in inlier_indices)

        # 1. Disegno Base: Marker + Linee Proiezione (con distinzione Inlier/Outlier)
        from .viz import draw_detected_markers_with_projection, draw_large_axes
        
        # Recuperiamo solo le detection e le pose effettivamente mappate (presenti in debug_info)
        mapped_detections = [m["detection"] for m in markers_debug_info]
        mapped_poses = [poses[valid_detections.index(m["detection"])] for m in markers_debug_info]

        debug_img = draw_detected_markers_with_projection(
            image, 
            mapped_detections, 
            mapped_poses, 
            K, 
            dist, 
            marker_size,
            projections=ghost_projections,
            inliers=is_inlier_list
        )
        
        # 2. SFERA MEDIA (Rosso trasparente) - Solo volume e wireframe
        rvec_sphere_cam, _ = cv.Rodrigues(T_cam_sphere_avg[:3, :3])
        debug_img = draw_sphere_overlay(
            debug_img, K, dist,
            rvec=rvec_sphere_cam,
            tvec=T_cam_sphere_avg[:3, 3],
            radius=GEO_RADIUS,
            color=(0, 0, 255),
            alpha=0.15,
            tvec_axes=None # Rimuoviamo assi dal centro sfera per pulizia
        )
        
        # 3. PUNTA PENNA (Assi Grandi)
        # Disegna assi molto visibili sulla punta
        debug_img = draw_large_axes(
            debug_img, K, dist,
            rvec=rvec_sphere_cam,   
            tvec=T_cam_tip[:3, 3],  
            scale=1 # Ingrandisce gli assi
        )
        
        # Info Text
        dist_cm = np.linalg.norm(T_cam_sphere_avg[:3, 3]) * 100
        label = f"Tip: {dist_cm:.1f}cm | Active: {len(sphere_candidates_cam)}"
        cv.putText(debug_img, label, (20, 50), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        overlay_img = debug_img

    return {
        # Output nel Frame Richiesto (Robot Base se T_base_cam è presente, altrimenti Camera)
        "rvec": rvec_final_base,
        "tvec": t_final_base,
        "R": R_final_base,
        "quat": quat_final_base,
        
        # Dati grezzi/debug
        "num_markers": len(sphere_candidates_cam),
        "overlay": overlay_img,
        
        # Extra: se serve sapere la posa punta rispetto alla camera (es. per servoing visivo diretto)
        "tvec_tip_cam": T_cam_tip[:3, 3]
    }