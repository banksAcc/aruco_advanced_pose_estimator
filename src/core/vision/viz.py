# pc/app/algo/viz.py
import cv2 as cv
import numpy as np
from typing import Tuple, List, Optional

def draw_sphere_wireframe(img: np.ndarray, K: np.ndarray, dist: np.ndarray, 
                          rvec: np.ndarray, tvec: np.ndarray, 
                          radius: float, 
                          color: Tuple[int, int, int] = (0, 0, 255),
                          rings: int = 12, 
                          segments: int = 24,
                          alpha: float = 0.5) -> np.ndarray: # <--- Nuovo parametro
    """
    Disegna una sfera wireframe con supporto alla trasparenza (alpha).
    """
    # 1. Crea un overlay vuoto (nero)
    overlay = np.zeros_like(img)
    h, w = img.shape[:2]
    
    # --- Generazione punti 3D (Meridiani e Paralleli) ---
    all_pts_3d = []
    # Paralleli
    for i in range(1, rings):
        phi = np.pi * i / rings
        z, r_phi = radius * np.cos(phi), radius * np.sin(phi)
        for j in range(segments):
            theta = 2 * np.pi * j / segments
            all_pts_3d.append([r_phi * np.cos(theta), r_phi * np.sin(theta), z])
    # Meridiani
    for i in range(rings):
        theta = np.pi * i / rings
        for j in range(segments):
            phi = 2 * np.pi * j / segments
            all_pts_3d.append([radius * np.sin(phi) * np.cos(theta), 
                               radius * np.sin(phi) * np.sin(theta), 
                               radius * np.cos(phi)])

    pts_3d = np.array(all_pts_3d, dtype=np.float32)
    pts_2d, _ = cv.projectPoints(pts_3d, rvec, tvec, K, dist)
    pts_2d = pts_2d.reshape(-1, segments, 2).astype(int)

    # 2. Disegna le linee sull'OVERLAY invece che sull'immagine finale
    for ring in pts_2d:
        for j in range(len(ring)):
            pt1, pt2 = tuple(ring[j]), tuple(ring[(j + 1) % len(ring)])
            if (0 <= pt1[0] < w and 0 <= pt1[1] < h) or (0 <= pt2[0] < w and 0 <= pt2[1] < h):
                cv.line(overlay, pt1, pt2, color, 1, cv.LINE_AA)

    # 3. Fonde l'overlay con l'immagine originale
    # output = img * (1 - alpha) + overlay * alpha
    output = cv.addWeighted(img, 1.0, overlay, alpha, 0)
    
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
