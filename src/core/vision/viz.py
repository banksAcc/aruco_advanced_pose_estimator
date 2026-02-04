# pc/app/algo/viz.py
import cv2 as cv
import numpy as np
from typing import Tuple, List, Optional

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
        
        """
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
        """

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

def draw_large_axes(img: np.ndarray, K: np.ndarray, dist: np.ndarray, 
                    rvec: np.ndarray, tvec: np.ndarray, scale: float = 1.0):
    """Disegna assi X,Y,Z molto visibili e spessi."""
    # Lunghezza assi: es. 0.05m * scale
    length = 0.05 * scale
    cv.drawFrameAxes(img, K, dist, rvec, tvec, length, 4) # Spessore 4
    return img

from typing import List, Optional, Union
import numpy as np
import cv2 as cv

def draw_detected_markers_with_projection(
    img: np.ndarray, 
    detections, 
    poses, 
    K, 
    dist, 
    marker_size,
    projections: List[Optional[np.ndarray]] = None,
    statuses: List[str] = None # Riceve "inlier", "outlier" o "tilted"
):
    """
    Disegna il feedback visivo dei marker con codifica a colori:
    - GREEN: Inlier (utilizzato per la media)
    - YELLOW: Tilted (scartato per inclinazione eccessiva)
    - RED: Outlier (scartato per distanza euclidea)
    """
    out = img.copy()
    n = len(detections)
    
    if projections is None:
        projections = [None] * n
    
    # Fallback se lo stato non è fornito
    if statuses is None:
        statuses = ["inlier"] * n

    for i, (det, pose, proj_tvec) in enumerate(zip(detections, poses, projections)):
        status = statuses[i]
        
        # --- DEFINIZIONE LOGICA COLORI ---
        if status == "inlier":
            color_main = (0, 255, 0)      # Verde
            color_ghost = (255, 255, 0)   # Ciano
            label = None
        elif status == "tilted":
            color_main = (0, 255, 255)    # Giallo
            color_ghost = (100, 100, 100) # Grigio
            label = "TILTED"
        else: # status == "outlier"
            color_main = (0, 0, 255)      # Rosso
            color_ghost = (100, 100, 100) # Grigio
            label = "OUTLIER"
        
        # 1. Box Marker
        pts = det.corners.reshape((-1, 1, 2)).astype(np.int32)
        cv.polylines(out, [pts], True, color_main, 2)
        
        # Calcolo centro 2D del marker per testi e linee
        marker_center_2d = np.mean(det.corners, axis=0).astype(int)
        cx_m, cy_m = marker_center_2d[0], marker_center_2d[1]
        
        # 2. Pallino sul Marker (Blu scuro per contrasto)
        cv.circle(out, (cx_m, cy_m), 4, (255, 0, 0), -1)

        # 3. Gestione Scritte (Label)
        if label:
            # Testo leggermente sopra il marker
            cv.putText(out, label, (cx_m - 20, cy_m - 15), 
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, color_main, 2, cv.LINE_AA)

        # 4. Proiezione del centro sfera (Ghost)
        if proj_tvec is not None:
            ghost_2d, _ = cv.projectPoints(proj_tvec.reshape(1, 1, 3), 
                                           np.zeros(3), np.zeros(3), 
                                           K, dist)
            gx, gy = ghost_2d[0].ravel().astype(int)
            
            # Linea di collegamento (Marker -> Ghost)
            # Linea tratteggiata simulata o spessore diverso per gli scarti
            thickness = 1 if status == "inlier" else 2
            cv.line(out, (cx_m, cy_m), (gx, gy), color_ghost, thickness, cv.LINE_AA)
            
            # Pallino sul Ghost Center
            cv.circle(out, (gx, gy), 4, color_ghost, -1)
            
    return out
