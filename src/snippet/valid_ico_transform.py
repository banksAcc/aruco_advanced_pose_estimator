
"""
Script di debug che utilizziamo per verificare la correttezza del json di rototrasalzione dei marker
dalla superficie del solido al centro del solido.

Per ogni marker trovato salva il frame nella cartella di debug/ID_xx con l'overlay degli assi del marker e del centro del solido.
"""

import cv2 as cv
import numpy as np
import os
from datetime import datetime
import sys
import pathlib
from pathlib import Path

# Trova la cartella 'src' (ovvero il nonno del file attuale)
root_path = pathlib.Path(__file__).parent.parent.resolve()
sys.path.append(str(root_path))

# Import dai tuoi moduli
from core.vision.detect import detect_markers
from core.vision.pnp import estimate_marker_poses
from utils.config_models import AppConfig
from utils.utils import load_camera_calibration, load_ico_transforms, load_config

OUTPUT_DIR = "debug"      # Cartella output immagini

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def draw_single_marker_debug(img, K, dist, m_pose, face_id, transforms, cfg: AppConfig):
    """
    Funzione helper che disegna su 'img' SOLO i dati relativi a questo marker.
    """
    # 1. Disegna Assi Marker Locali (Piccoli)
    cv.drawFrameAxes(img, K, dist, m_pose.rvec, m_pose.tvec, cfg.pose.ico.marker_size_mm/1000, 2)
    
    # 2. Calcola e Disegna Centro Solido Stimato
    if face_id and face_id in transforms:
        T_face_body = transforms[face_id]

        # Matrice Marker -> Camera
        T_cam_marker = np.eye(4)
        T_cam_marker[:3, :3] = m_pose.R
        T_cam_marker[:3, 3] = m_pose.tvec.flatten()

        # Matrice Body -> Camera (Secondo QUESTO marker)
        T_cam_body = T_cam_marker @ T_face_body

        # Estrai vettori per disegnare
        rvec_body, _ = cv.Rodrigues(T_cam_body[:3, :3])
        t_body = T_cam_body[:3, 3]

        # Disegna Assi Corpo (Grandi e Spessi)
        # Se la matrice è giusta, questo sistema deve essere al centro della sfera
        cv.drawFrameAxes(img, K, dist, rvec_body, t_body, cfg.pose.ico.marker_size_mm/1000 * 3, 3)
        
        # Linea Gialla: Marker -> Centro Stimato
        marker_center_px, _ = cv.projectPoints(m_pose.tvec, np.zeros(3), np.zeros(3), K, dist)
        body_center_px, _ = cv.projectPoints(t_body, np.zeros(3), np.zeros(3), K, dist)
        
        p1 = tuple(marker_center_px[0].ravel().astype(int))
        p2 = tuple(body_center_px[0].ravel().astype(int))
        
        cv.line(img, p1, p2, (0, 255, 255), 2)
        
        # Info testuali
        info = f"ID:{face_id} Dist:{np.linalg.norm(t_body)*100:.1f}cm"
        cv.putText(img, info, (p1[0], p1[1]-15), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

def main():
    print("[INFO] Caricamento dati...")
    cfg = load_config(Path("../config/config.yaml"))
    
    try:
        K, dist = load_camera_calibration(cfg.pose.camera_calibration_npz)
        transforms = load_ico_transforms(cfg.pose.ico.transform_file)
    except Exception as e:
        print(f"[ERROR] {e}")
        return

    cap = cv.VideoCapture(0)
    # Imposta risoluzione (opzionale)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("[ERROR] Webcam non trovata.")
        return

    ensure_dir(OUTPUT_DIR)
    print(f"[INFO] Saving isolated debug frames to: {OUTPUT_DIR}/ID_xx/")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Immagine per la visualizzazione a schermo (Live View)
        # Qui accumuliamo TUTTI i disegni per comodità dell'utente
        live_img = frame.copy()

        detections = detect_markers(frame, cfg.pose.ico.dictionary)

        if detections:
            poses = estimate_marker_poses(detections, K, dist, cfg.pose.ico.marker_size_mm/1000)
            
            for det, m_pose in zip(detections, poses):
                face_id = cfg.marker_map.get(det.id)
                
                # --- A. Disegno sulla Live View (Accumulativo) ---
                draw_single_marker_debug(live_img, K, dist, m_pose, face_id, transforms, cfg)

                # --- B. Salvataggio Isolato (Esclusivo per questo ID) ---
                # Creiamo una copia PULITA del frame originale
                save_img = frame.copy()
                
                # Disegniamo SOLO le info di questo marker
                draw_single_marker_debug(save_img, K, dist, m_pose, face_id, transforms, cfg)
                
                # Salviamo nella cartella specifica
                id_dir = os.path.join(OUTPUT_DIR, f"ID_{det.id}")
                ensure_dir(id_dir)
                
                ts = datetime.now().strftime("%H%M%S_%f")
                filename = os.path.join(id_dir, f"iso_{ts}.jpg")
                cv.imwrite(filename, save_img)

        cv.imshow("Live Debug (All Markers)", live_img)
        
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()