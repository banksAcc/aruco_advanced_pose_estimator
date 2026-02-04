
"""
Codice per test veloce della funzione di stima della posa dell'icosaedro estimate_truncated_ico_from_image
Utilizziamo la webcam per il test e il json di rototraslazione per il solido.

Configurazione tramite Variabili Globali.
"""

import sys
import pathlib
from pathlib import Path
import cv2 as cv2
from typing import Optional
import numpy as np

# Trova la cartella 'src' (ovvero il nonno del file attuale)
root_path = pathlib.Path(__file__).parent.parent.resolve()
sys.path.append(str(root_path))

from core.vision.api import estimate_truncated_ico_from_image
from utils.utils import load_camera_calibration, load_ico_transforms, load_config
from core.robot.robot_transform import get_base_to_camera_matrix

def main() -> int:
    # 1. Caricamento Risorse
    print(f"Caricamento configurazione da ../config/config.yaml...")
    cfg = load_config(Path("../config/config.yaml"))
    
    calib_path = cfg.pose.camera_calibration_npz
    trans_path = cfg.pose.ico.transform_file
    
    # Caricamento K, dist
    K, dist = load_camera_calibration(str(calib_path))
    
    # Caricamento e SCALING delle trasformazioni
    # Le trasformazioni nel JSON sono unitarie (raggio=1). 
    # Le moltiplichiamo qui per il raggio reale.
    transforms_raw = load_ico_transforms(str(trans_path))

    # Parametri marker dal config
    marker_size_m = float(cfg.pose.ico.marker_size_mm) / 1000.0
    dictionary = cfg.pose.ico.dictionary

    T_base_cam: Optional[np.ndarray] = get_base_to_camera_matrix(
        cfg.pose.extrinsic_calibration.trans_x_mm,
        cfg.pose.extrinsic_calibration.trans_y_mm,
        cfg.pose.extrinsic_calibration.trans_z_mm,
        cfg.pose.extrinsic_calibration.rot_phi_deg,
        cfg.pose.extrinsic_calibration.rot_theta_deg,
        cfg.pose.extrinsic_calibration.rot_psi_deg
    )
    
    # 2. Setup Camera
    cap = cv2.VideoCapture(int(cfg.capture.camera_id))
    
    if not cap.isOpened():
        print("Errore apertura camera.")
        return 1

    print("Tracking avviato. Premi 'q' per uscire.")
    
    while True:
        ret, frame = cap.read()
        if not ret: break

        # 3. Loop Principale
        try:
            result = estimate_truncated_ico_from_image(
                image=frame,
                K=K,
                dist=dist,
                transforms=transforms_raw,
                aruco_dict=dictionary,
                marker_size=marker_size_m,
                return_overlay=True,
                cfg=cfg,
                T_base_cam=T_base_cam,
            )
            
            final_img = result.get("overlay", frame)
            
        except ValueError:
            # Errori attesi (nessun marker, etc)
            final_img = frame
        except Exception as e:
            print(f"Errore runtime: {e}")
            final_img = frame

        cv2.imshow("Ico Tracker Test", final_img)
        if cv2.waitKey(200) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return 0

if __name__ == "__main__":
    sys.exit(main())