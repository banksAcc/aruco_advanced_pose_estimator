# pc/app/algo/viz.py
import cv2 as cv
import numpy as np
from typing import Tuple, List, Optional

# Cache per i punti della sfera (evita di ricalcolarli a ogni frame)
_SPHERE_CACHE = None

def _get_sphere_geometry(radius: float, rings: int = 10, sectors: int = 10):
    global _SPHERE_CACHE
    if _SPHERE_CACHE is None or _SPHERE_CACHE[0] != radius:
        points = []
        lines_idx = []
        for i in range(rings + 1):
            theta = i * np.pi / rings 
            for j in range(sectors):
                phi = j * 2 * np.pi / sectors 
                x = radius * np.sin(theta) * np.cos(phi)
                y = radius * np.sin(theta) * np.sin(phi)
                z = radius * np.cos(theta)
                points.append([x, y, z])
        points = np.array(points, dtype=np.float32)
        for i in range(rings + 1):
            base = i * sectors
            for j in range(sectors):
                lines_idx.append((base + j, base + (j + 1) % sectors))
        for j in range(sectors):
            for i in range(rings):
                lines_idx.append((i * sectors + j, (i + 1) * sectors + j))
        _SPHERE_CACHE = (radius, points, lines_idx)
    return _SPHERE_CACHE[1], _SPHERE_CACHE[2]

def draw_sphere_overlay(img: np.ndarray, K: np.ndarray, dist: np.ndarray, 
                        rvec: np.ndarray, tvec: np.ndarray, 
                        radius: float, 
                        color: Tuple[int, int, int] = (0, 0, 255), 
                        alpha: float = 0.2,
                        tvec_axes: Optional[np.ndarray] = None) -> np.ndarray: # <--- NUOVO PARAMETRO
    """
    Disegna la sfera su 'tvec' e gli assi su 'tvec_axes' (se fornito, altrimenti su tvec).
    """
    overlay = img.copy()
    output = img.copy()
    
    # --- Disegno SFERA (usa tvec originale) ---
    center_2d_pts, _ = cv.projectPoints(np.array([[0.,0.,0.]]), rvec, tvec, K, dist)
    cx, cy = center_2d_pts[0].ravel().astype(int)
    
    edge_2d_pts, _ = cv.projectPoints(np.array([[radius, 0., 0.]]), rvec, tvec, K, dist)
    ex, ey = edge_2d_pts[0].ravel().astype(int)
    radius_px = int(np.linalg.norm([ex - cx, ey - cy]))
    
    h, w = img.shape[:2]
    if radius_px > 0 and 0 <= cx < w and 0 <= cy < h:
        # Sfera solida trasparente
        cv.circle(overlay, (cx, cy), radius_px, color, -1)
        cv.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)
        
        # Wireframe Sfera
        pts_3d, lines_idx = _get_sphere_geometry(radius)
        pts_2d, _ = cv.projectPoints(pts_3d, rvec, tvec, K, dist)
        pts_2d = pts_2d.reshape(-1, 2).astype(int)
        line_color = (color[0], color[1] + 100, color[2]) 
        
        for p1_idx, p2_idx in lines_idx:
            pt1 = tuple(pts_2d[p1_idx])
            pt2 = tuple(pts_2d[p2_idx])
            if (0 <= pt1[0] < w and 0 <= pt1[1] < h) or (0 <= pt2[0] < w and 0 <= pt2[1] < h):
                cv.line(output, pt1, pt2, line_color, 1, cv.LINE_AA)

        # --- Disegno ASSI (usa tvec_axes se esiste) ---
        target_tvec = tvec_axes if tvec_axes is not None else tvec
        
        # Lunghezza assi visualizzati
        axis_len = radius * 1.5
        cv.drawFrameAxes(output, K, dist, rvec, target_tvec, axis_len, 3)

    return output

def draw_detected_markers(img: np.ndarray, detections, poses, K, dist, size, 
                          red_centers: Optional[List[Optional[np.ndarray]]] = None,
                          green_centers: Optional[List[Optional[np.ndarray]]] = None):
    """
    Disegna i box dei marker e i pallini sui centri stimati, MA NON GLI ASSI LOCALI.
    """
    out = img.copy()
    n = len(detections)
    
    if red_centers is None: red_centers = [None] * n
    if green_centers is None: green_centers = [None] * n

    for i, (det, pose) in enumerate(zip(detections, poses)):
        # --- 1. Disegna Box del Marker ---
        pts = det.corners.reshape((-1, 1, 2)).astype(np.int32)
        cv.polylines(out, [pts], True, (0, 255, 255), 2)
        
        # --- MODIFICA: RIMOSSO cv.drawFrameAxes SUL MARKER ---
        
        # ID Marker
        c = det.corners[0]
        x, y = int(c[0]), int(c[1])
        cv.rectangle(out, (x, y - 5), (x, y + 5), (0, 0, 0), -1)
        
        # --- Helper: Disegna Pallino (SENZA ASSI) ---
        def draw_center_point(center_tvec, color_bgr):
             # Proiezione del punto 3D per disegnare il pallino
            pts_2d, _ = cv.projectPoints(center_tvec.reshape(1, 1, 3), 
                                         np.zeros(3), np.zeros(3), 
                                         K, dist)
            cx, cy = pts_2d[0].ravel().astype(int)
            
            # 1. Pallino colorato + Bordo bianco
            cv.circle(out, (cx, cy), 5, color_bgr, -1)
            cv.circle(out, (cx, cy), 6, (255, 255, 255), 1)
            
            # 2. Linea dal marker al centro stimato (visualizza il raggio)
            marker_center_2d = np.mean(det.corners, axis=0).astype(int)
            cv.line(out, tuple(marker_center_2d), (cx, cy), color_bgr, 1, cv.LINE_AA)
            
            # --- MODIFICA: RIMOSSO cv.drawFrameAxes NEL CENTRO SFERA ---

        # --- Esecuzione disegno per i centri ---
        
        # Disegna ROSSO (Normale)
        if red_centers[i] is not None:
            draw_center_point(red_centers[i], (0, 0, 255))
            
        # Disegna VERDE (Flippato)
        if green_centers[i] is not None:
            draw_center_point(green_centers[i], (0, 255, 0))

    return out


# pc/app/algo/viz.py
import cv2 as cv
import numpy as np
from typing import Tuple, List, Optional

# ... _get_sphere_geometry e draw_sphere_overlay rimangono invariati ...
# ... (assicurati solo che draw_sphere_overlay restituisca l'immagine modificata) ...

def draw_large_axes(img: np.ndarray, K: np.ndarray, dist: np.ndarray, 
                    rvec: np.ndarray, tvec: np.ndarray, scale: float = 1.0):
    """Disegna assi X,Y,Z molto visibili e spessi."""
    # Lunghezza assi: es. 0.05m * scale
    length = 0.05 * scale
    cv.drawFrameAxes(img, K, dist, rvec, tvec, length, 4) # Spessore 4
    return img

def draw_detected_markers_with_projection(
    img: np.ndarray, 
    detections, 
    poses, 
    K, 
    dist, 
    marker_size,
    projections: List[Optional[np.ndarray]] = None,
    inliers: List[bool] = None # <-- Nuovo parametro
):
    """
    Disegna:
    1. Box del marker (Verde se inlier, Rosso se outlier)
    2. Centro del marker (Punto Blu)
    3. Linea di collegamento: Centro Marker -> Centro Sfera Proiettato (Ghost)
    4. Centro Sfera Proiettato (Pallino Ciano se inlier, Grigio se outlier)
    
    NON disegna assi RGB sui marker.
    """
    out = img.copy()
    if projections is None:
        projections = [None] * len(detections)
    
    # Se non fornito, assumiamo siano tutti inlier per default
    if inliers is None:
        inliers = [True] * len(detections)

    for i, (det, pose, proj_tvec) in enumerate(zip(detections, poses, projections)):
        # Determina il colore in base allo stato di inlier
        is_inlier = inliers[i]
        # Verde per inlier, Rosso per outlier
        color_main = (0, 255, 0) if is_inlier else (0, 0, 255)
        # Ciano per inlier, Grigio scuro per outlier
        color_ghost = (255, 255, 0) if is_inlier else (100, 100, 100)
        
        # 1. Box Marker
        pts = det.corners.reshape((-1, 1, 2)).astype(np.int32)
        cv.polylines(out, [pts], True, color_main, 2)
        
        # Calcolo centro 2D del marker
        marker_center_2d = np.mean(det.corners, axis=0).astype(int)
        cx_m, cy_m = marker_center_2d[0], marker_center_2d[1]
        
        # 2. Pallino sul Marker (Blu scuro)
        cv.circle(out, (cx_m, cy_m), 4, (255, 0, 0), -1)

        # Se abbiamo la proiezione del centro sfera (Ghost)
        if proj_tvec is not None:
            ghost_2d, _ = cv.projectPoints(proj_tvec.reshape(1, 1, 3), 
                                           np.zeros(3), np.zeros(3), 
                                           K, dist)
            gx, gy = ghost_2d[0].ravel().astype(int)
            
            # 3. Linea di collegamento (Marker -> Ghost)
            # Usiamo uno spessore maggiore per gli outlier per evidenziarli
            thickness = 1 if is_inlier else 2
            cv.line(out, (cx_m, cy_m), (gx, gy), color_ghost, thickness, cv.LINE_AA)
            
            # 4. Pallino sul Ghost Center
            cv.circle(out, (gx, gy), 4, color_ghost, -1)
            
            # Opzionale: Aggiungi etichetta "OUTLIER" se scartato
            if not is_inlier:
                cv.putText(out, "OUTLIER", (cx_m, cy_m - 10), 
                           cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
    return out
