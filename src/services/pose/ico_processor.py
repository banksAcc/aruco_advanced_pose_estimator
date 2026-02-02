from __future__ import annotations
from typing import Any, Optional, Tuple, TYPE_CHECKING
import numpy as np

# Imports specifici del dominio
from core.vision.api import estimate_truncated_ico_from_image
from utils.utils import load_camera_calibration, load_ico_transforms
from core.robot.robot_transform import transform_camera_to_robot

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

        #FIXME: SIAMO SICURI CHE LA CONVERSIONE è CORRETTA ?  
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
        
        # A. Calibrazione Camera
        if self._cached_calib_path != str(calib_path):
            try:
                self._K, self._dist = load_camera_calibration(str(calib_path))
                self._cached_calib_path = str(calib_path)
            except Exception as e:
                return ({"file": packet.filename, "ok": False, "reason": f"calib_load_err: {e}"}, None)
        
        #TODO: descrivere il processo di scalig, cosa stiamo facendo e a che serve
        # B. Trasformazioni Faccia->Corpo (CORRETTO: NESSUNO SCALING)
        if self._cached_trans_path != str(transform_path):
            try:
                # Carichiamo le trasformazioni pure. Assumiamo che il JSON sia
                # già corretto e in METRI.
                self._transforms = load_ico_transforms(str(transform_path))
                
                # --- DEBUG ---
                # Verifica immediata per evitare dubbi futuri
                first_key = next(iter(self._transforms))
                t_sample = self._transforms[first_key][:3, 3]
                print(f"[ICO-LOAD] Loaded transforms. Sample ({first_key}) norm: {np.linalg.norm(t_sample):.4f}m")
                # Se leggi ~0.058m qui, è PERFETTO.
                # -------------

                self._cached_trans_path = str(transform_path)
            except Exception as e:
                 return ({"file": packet.filename, "ok": False, "reason": f"trans_load_err: {e}"}, None)

        # --- 2. STIMA POSA ---
        try:
            result = estimate_truncated_ico_from_image(
                image=packet.frame,
                K=self._K,
                dist=self._dist,
                transforms=self._transforms,
                aruco_dict=dict_name,
                marker_size=marker_size,
                return_overlay=True,
            )
            
        except ValueError as e:
            return (
                {"file": packet.filename, "ok": False, "reason": f"pose_fail:  {str(e)}"},
                None,
            )
        except Exception as e:
            # Ritorna stato fallito ma non crashare
            return (
                {"file": packet.filename, "ok": False, "reason": f"pose_fail: {str(e)}"},
                None,
            )
    
        # --- 2.5 TRASFORMAZIONE ROBOT ---
        rvec_base, tvec_base = None, None

        if result.get("ok", False):
            tvec_tip_cam = result.get("tvec_tip")
            rvec_tip_cam = result.get("rvec")
            
            if tvec_tip_cam is not None and rvec_tip_cam is not None:
                rvec_base, tvec_base = transform_camera_to_robot(
                    rvec_tip_cam, 
                    tvec_tip_cam, 
                    self._T_base_cam
                )

        # --- 3. FORMATTAZIONE OUTPUT ---
        overlay = result.get("overlay")

        # Flattening sicuro per JSON
        rvec_flat = result["rvec"].flatten().tolist()
        tvec_flat = result["tvec"].flatten().tolist()

        frame_entry: dict[str, Any] = {
            "file": packet.filename,
            "ok": True,
            # Dati RAW Camera
            "rvec": rvec_flat,
            "tvec": tvec_flat,
            # Dati TRASFORMATI
            "rvec_robot": rvec_base.tolist() if rvec_base is not None else None,
            "tvec_robot": tvec_base.tolist() if tvec_base is not None else None,
            
            "reproj_err": None,
            "num_markers": int(result.get("num_markers", 0)),
            "timestamp": packet.iso_timestamp,
        }

        if "filter_debug" in result and result["filter_debug"]:
            frame_entry["marker_filter"] = result["filter_debug"]

        return frame_entry, overlay