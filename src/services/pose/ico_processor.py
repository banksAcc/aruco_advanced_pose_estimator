from __future__ import annotations
from typing import Any, Optional, Tuple, TYPE_CHECKING
import numpy as np
import cv2 as cv # Aggiunto per convertire R in rvec se necessario

# Imports specifici del dominio
from core.vision.api import estimate_truncated_ico_from_image
from utils.utils import load_camera_calibration, load_ico_transforms

if TYPE_CHECKING:
    from utils.config_models import FramePacket
    from utils.config_models import AppConfig

class IcoPoseProcessor:
    """
    Gestisce la logica specifica per la stima della posa tramite Icosaedro.
    Mantiene in cache le calibrazioni e le trasformazioni per efficienza.
    """

    def __init__(self, cfg: AppConfig, base_to_cam_matrix: Optional[np.ndarray]):
        self.cfg = cfg
        self.pose_cfg = cfg.pose.ico
        self._T_base_cam = base_to_cam_matrix

        # Cache interna per evitare reload continui
        self._cached_calib_path: Optional[str] = None
        self._K: Optional[np.ndarray] = None
        self._dist: Optional[np.ndarray] = None
        
        self._cached_trans_path: Optional[str] = None
        self._transforms: Optional[dict] = None

    def process(self, packet: FramePacket) -> Tuple[dict[str, Any], Optional[np.ndarray]]:
        """
        Elabora un singolo frame packet e restituisce il risultato JSON e l'overlay.
        """
        pose_cfg = self.pose_cfg
        dict_name = pose_cfg.dictionary
 
        # Conversione mm -> metri
        marker_size = float(pose_cfg.marker_size_mm) / 1000.0 
        
        # Percorsi file
        transform_path = pose_cfg.transform_file
        calib_path = self.cfg.pose.camera_calibration_npz

        # Controllo path calibrazione
        if not calib_path:
            return (
                {"file": packet.filename, "ok": False, "reason": "no_calibration"},
                None,
            )

        # --- 1. CARICAMENTO E CACHING RISORSE (K, Dist, Transforms) ---
        if self._cached_calib_path != str(calib_path):
            try:
                self._K, self._dist = load_camera_calibration(str(calib_path))
                self._cached_calib_path = str(calib_path)
            except Exception as e:
                return ({"file": packet.filename, "ok": False, "reason": f"calib_load_err: {e}"}, None)
        
        if self._cached_trans_path != str(transform_path):
            try:
                self._transforms = load_ico_transforms(str(transform_path))
                self._cached_trans_path = str(transform_path)
            except Exception as e:
                 return ({"file": packet.filename, "ok": False, "reason": f"trans_load_err: {e}"}, None)
             
        # --- 2. STIMA POSA ---
        try:
            # La funzione ora accetta T_base_cam e gestisce internamente la trasformazione
            result = estimate_truncated_ico_from_image(
                image=packet.frame,
                K=self._K,
                dist=self._dist,
                transforms=self._transforms,
                aruco_dict=dict_name,
                marker_size=marker_size,
                cfg=self.cfg,
                T_base_cam=self._T_base_cam,
                return_overlay=True
            )
            
        except ValueError as e:
            return (
                {"file": packet.filename, "ok": False, "reason": f"pose_fail:  {str(e)}"},
                None,
            )
        except Exception as e:
            return (
                {"file": packet.filename, "ok": False, "reason": f"pose_fail: {str(e)}"},
                None,
            )


        # --- 3. ESTRAZIONE DATI E FORMATTAZIONE ---
        # Estrazione dati Posa Camera
        tip_cam = result["tip_in_camera"]
        # Convertiamo R in rvec (Rodrigues) se non presente nel dizionario, per coerenza con l'output JSON precedente
        rvec_cam, _ = cv.Rodrigues(tip_cam["R"])
        tvec_cam = tip_cam["tvec"]

        # Estrazione dati Posa Robot (se disponibile)
        tip_robot = result.get("tip_in_robot")
        rvec_robot_list = None
        tvec_robot_list = None

        if tip_robot is not None:
             rvec_rob, _ = cv.Rodrigues(tip_robot["R"])
             rvec_robot_list = rvec_rob.flatten().tolist()
             tvec_robot_list = tip_robot["tvec"].flatten().tolist()

        # Estrazione Statistiche
        stats = result.get("markers_stats", {})

        # Estrazione Overlay
        overlay = result.get("overlay")

        frame_entry: dict[str, Any] = {
            "file": packet.filename,
            "ok": True,
            
            # --- Output Base Camera ---
            "rvec": rvec_cam.flatten().tolist(),
            "tvec": tvec_cam.flatten().tolist(),
            "dist_cam": float(tip_cam.get("dist", 0.0)), # Opzionale: Distanza punta-camera
            
            # --- Output Base Robot ---
            "rvec_robot": rvec_robot_list,
            "tvec_robot": tvec_robot_list,
            "dist_robot": float(tip_robot.get("dist", 0.0)) if tip_robot else None,
            
            # --- Statistiche ---
            "reproj_err": None, # Non calcolato globalmente in questo algoritmo
            "num_markers": stats.get("found_valid", 0),
            "num_outliers": stats.get("discarded_outliers", 0),
            
            "timestamp": packet.iso_timestamp,
        }

        # Aggiungiamo eventuali debug extra se presenti
        if "markers_debug_info" in result:
             # Nota: serializzare markers_debug_info potrebbe essere pesante per il JSON, 
             # valutare se includerlo solo in modalità debug
             pass

        return frame_entry, overlay
