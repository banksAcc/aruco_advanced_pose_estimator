"""Asynchronous worker that computes pose from captured frames."""

from __future__ import annotations

import asyncio
import csv
import json
from concurrent.futures import ThreadPoolExecutor
from utils.config_models import SessionJob
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, TYPE_CHECKING
import numpy as np
import cv2

# Importa il nuovo modulo robotico
from core.robot.robot_transform import get_base_to_camera_matrix

from utils.config_models import (
    AppConfig,
    PoseEndMessage,
    PoseStartMessage,
    PoseWorkerPayload,
)

from utils.logger import get_logger
from utils.utils import (
    Encode_as_bytes,
)

from utils.config_models import FramePacket
from .ico_processor import IcoPoseProcessor

log = get_logger("POSE")

class PoseWorker:
    """Asynchronous worker that estimates ico pose for capture sessions."""

    def __init__(
        self,
        cfg: AppConfig,
        output_root: Path,
        ble_queue: asyncio.Queue[Optional[Encode_as_bytes]],
        
    ):
        self.cfg = cfg
        self.output_root = output_root
        self.queue: asyncio.Queue[Optional[PoseWorkerPayload]] = asyncio.Queue()
        self.tasks: list[asyncio.Task] = []
        self.max_jobs = int(cfg.pose.max_parallel_jobs)
        self.enabled = bool(cfg.pose.enabled)
        self.ble_queue = ble_queue
        try:
            self.loop = asyncio.get_running_loop()
        except RuntimeError:
            self.loop = asyncio.get_event_loop()

        self.sessions: Dict[str, SessionJob] = {}
        self.method = cfg.pose.method.lower()
        self.pose_cfg_ico = cfg.pose.ico
        self.save_overlay = bool(cfg.pose.save_overlay)
        self.save_executor = ThreadPoolExecutor(max_workers=1)
        
        self._T_base_cam: Optional[np.ndarray] = get_base_to_camera_matrix(
            cfg.pose.extrinsic_calibration.trans_x_mm,
            cfg.pose.extrinsic_calibration.trans_y_mm,
            cfg.pose.extrinsic_calibration.trans_z_mm,
            cfg.pose.extrinsic_calibration.rot_phi_deg,
            cfg.pose.extrinsic_calibration.rot_theta_deg,
            cfg.pose.extrinsic_calibration.rot_psi_deg
        )

        # Inizializziamo il processore delegato
        self.ico_processor = IcoPoseProcessor(cfg, self._T_base_cam)

    async def start(self) -> None:
        """Spawn worker tasks if pose estimation is enabled."""
        if not self.enabled:
            log.info("disabled")
            return
        log.info(f"starting workers = {self.max_jobs}")
        for _ in range(self.max_jobs):
            self.tasks.append(asyncio.create_task(self._worker()))

    async def stop(self) -> None:
        """Signal workers to exit and wait for completion."""
        for job in list(self.sessions.values()):
            job.finished.set()
            try:
                job.frame_queue.put_nowait(None)
            except asyncio.QueueFull:
                pass
        for _ in self.tasks:
            await self.queue.put(None)
        await asyncio.gather(*self.tasks, return_exceptions=True)
        self.tasks.clear()
        await asyncio.gather(
            *(job.task for job in list(self.sessions.values()) if job.task),
            return_exceptions=True,
        )
        self.sessions.clear()
        self.save_executor.shutdown(wait=True)

    async def _worker(self) -> None:
        """Consume jobs from the queue and dispatch session handling."""
        while True:
            job = await self.queue.get()
            if job is None:
                break
            if isinstance(job, PoseStartMessage):
                self._handle_start(job)
            elif isinstance(job, PoseEndMessage):
                self._handle_end(job)
            else:
                log.warning(f"Unknown pose job payload: {job!r}")

    def _handle_start(self, payload: PoseStartMessage) -> None:
        session_key = payload.session_key
        if session_key in self.sessions:
            log.warning(f"Session {session_key} already tracked -> ignoring start")
            return

        session_dir = payload.session_dir
        frame_queue = payload.frame_queue
        freq_ms = int(payload.freq_ms)
        label = payload.label or session_dir.name
        save_frames = bool(payload.save_frames)
        save_dir = payload.save_dir
        results: dict[str, Any] = {
            "session": label,
            "start": payload.start,
            "end": None,
            "frequency_ms": freq_ms,
            "method": self.method,
            "frames": [],
        }
        job = SessionJob(
            key=session_key,
            frame_queue=frame_queue,
            freq_ms=freq_ms,
            start_iso=payload.start,
            results=results,
            label=label,
            save_frames=save_frames,
            save_dir=save_dir,
            save_overlay=self.save_overlay,
        )
        job.task = asyncio.create_task(self._run_session(job))
        self.sessions[session_key] = job
        log.info(f"Pose session started for {label}")

    def _handle_end(self, payload: PoseEndMessage) -> None:
        session_key = payload.session_key
        job = self.sessions.get(session_key)
        if job is None:
            log.warning(f"Pose END for unknown session {session_key}")
            return
        if payload.save_dir:
            job.save_dir = payload.save_dir
        job.end_iso = payload.end
        job.results["end"] = job.end_iso
        job.finished.set()
        log.info(f"Pose session finishing for {job.label}")

    async def _run_session(self, job: SessionJob) -> None:
        try:
            await self._consume_session(job)
        finally:
            self.sessions.pop(job.key, None)

    async def _consume_session(self, job: SessionJob) -> None:
        self._notify_ble(Encode_as_bytes(self.cfg.ble.message_at_computation_start))
        loop = asyncio.get_running_loop()
        try:
            await self._process_session_stream(job, loop)

            if not job.finished.is_set():
                await job.finished.wait()

            if job.end_iso is None:
                job.results["end"] = job.results.get("end") or job.start_iso
        finally:
            self._write_results(job)
            self._cleanup_frames(job)
            self._notify_ble(Encode_as_bytes(self.cfg.ble.message_at_computation_end))

    async def _process_session_stream(
        self, job: SessionJob, loop: asyncio.AbstractEventLoop
    ) -> None:
        while True:
            packet = await job.frame_queue.get()
            if packet is None:
                break

            frame_result, overlay = await asyncio.to_thread(
                self._process_frame_packet, job, packet
            )
            job.results["frames"].append(frame_result)

            overlay_path = None
            if overlay is not None and job.save_overlay:
                overlay_path = self._derive_overlay_path(job, packet)
                if overlay_path is not None:
                    job.overlay_paths.append(overlay_path)
                    frame_result["overlay_file"] = overlay_path.name

            if job.save_frames:
                await self._save_packet(job, packet, overlay, overlay_path, loop)

            packet.frame = None  # release reference as soon as possible

    async def _save_packet(
        self,
        job: SessionJob,
        packet: FramePacket,
        overlay: Any,
        overlay_path: Optional[Path],
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        image = packet.frame
        if image is None and overlay is None:
            return

        path = packet.save_path
        if path is None:
            if job.save_dir is None:
                return
            path = Path(job.save_dir) / packet.filename

        def _write() -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            try:
                if image is not None:
                    ok = cv2.imwrite(str(path), image)
                    if not ok:
                        log.warning(f"cv2.imwrite returned False for {path.name}")
                if overlay is not None and overlay_path is not None:
                    overlay_path.parent.mkdir(parents=True, exist_ok=True)
                    ok_overlay = cv2.imwrite(str(overlay_path), overlay)
                    if not ok_overlay:
                        log.warning(
                            f"cv2.imwrite returned False for {overlay_path.name}"
                        )
            except Exception as exc:
                log.warning(f"Failed to write frame {path.name}: {exc}")
                return

        try:
            await loop.run_in_executor(self.save_executor, _write)
        except Exception as exc:
            log.warning(f"Failed to persist frame {path.name}: {exc}")

    def _derive_overlay_path(
        self, job: SessionJob, packet: FramePacket
    ) -> Optional[Path]:
        base_path = packet.save_path
        if base_path is None:
            if job.save_dir is None:
                return None
            base_path = Path(job.save_dir) / packet.filename
        stem = base_path.stem
        suffix = base_path.suffix
        return base_path.with_name(f"{stem}_overlay{suffix}")

    def _cleanup_frames(self, job: SessionJob) -> None:
        if not job.save_frames:
            job.overlay_paths.clear()
            return
        if job.save_overlay:
            job.overlay_paths.clear()
            return
        directory = job.save_dir
        if directory is None:
            job.overlay_paths.clear()
            return
        try:
            for overlay_path in directory.glob("*_overlay.*"):
                try:
                    overlay_path.unlink()
                except FileNotFoundError:
                    continue
                except OSError as exc:
                    log.debug(f"Could not remove {overlay_path.name}: {exc}")
        except OSError as exc:
            log.debug(f"Overlay cleanup failed in {directory}: {exc}")
        job.overlay_paths.clear()

    def _process_frame_packet(
            self, job: SessionJob, packet: FramePacket
        ) -> Tuple[dict[str, Any], Optional["np.ndarray"]]:
            
            if self.method == "ico":
                # ORA CHIAMIAMO IL NUOVO PROCESSOR
                return self.ico_processor.process(packet)
            
            return (
                {
                    "file": packet.filename,
                    "ok": False,
                    "reason": f"invalid_method_{self.method}",
                },
                None,
            )
    
    def _write_results(self, job: SessionJob) -> None:
        label = job.label
        
        # 1. JSON (Manteniamo il dump completo per sicurezza/debug futuro)
        out_json = self.output_root / f"{label}_pose.json"
        try:
            with out_json.open("w", encoding="utf-8") as f:
                json.dump(job.results, f, indent=2)
            log.info(f"Pose results written to {out_json.name}")
        except Exception as exc:
            log.error(f"Failed to write pose results: {exc}")

        # 2. CSV COMPLETO (Tutti i dati dei frame)
        out_csv = self.output_root / f"{label}_pose.csv"
        try:
            with out_csv.open("w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                
                # HEADER COMPLETO
                headers = [
                    "frame_index", "timestamp", "file", "ok", 
                    "num_markers", "num_outliers", "reproj_err",
                    "dist_cam", "dist_robot",
                    "tx_cam", "ty_cam", "tz_cam", 
                    "rx_cam", "ry_cam", "rz_cam",
                    "tx_robot", "ty_robot", "tz_robot", 
                    "rx_robot", "ry_robot", "rz_robot",
                    "overlay_file"
                ]
                writer.writerow(headers)

                for idx, frame in enumerate(job.results.get("frames", []), start=1):
                    # Estrazione dati con fallback sicuri
                    t_cam = frame.get("tvec") or [None]*3
                    r_cam = frame.get("rvec") or [None]*3
                    t_rob = frame.get("tvec_robot") or [None]*3
                    r_rob = frame.get("rvec_robot") or [None]*3

                    # Helper per formattazione numerica
                    def fmt(val, precision=6):
                        if val is None: return ""
                        try:
                            return f"{val:.{precision}f}"
                        except (ValueError, TypeError):
                            return str(val)

                    row = [
                        idx,
                        frame.get("timestamp", ""),
                        frame.get("file", ""),
                        frame.get("ok", False),
                        frame.get("num_markers", 0),
                        frame.get("num_outliers", 0),
                        fmt(frame.get("reproj_err")),
                        fmt(frame.get("dist_cam")),
                        fmt(frame.get("dist_robot")),
                        # Pose Camera
                        fmt(t_cam[0]), fmt(t_cam[1]), fmt(t_cam[2]),
                        fmt(r_cam[0]), fmt(r_cam[1]), fmt(r_cam[2]),
                        # Pose Robot
                        fmt(t_rob[0]), fmt(t_rob[1]), fmt(t_rob[2]),
                        fmt(r_rob[0]), fmt(r_rob[1]), fmt(r_rob[2]),
                        # File di overlay
                        frame.get("overlay_file", "")
                    ]
                    writer.writerow(row)

            log.info(f"Complete Pose CSV written to {out_csv.name}")
        except Exception as exc:
            log.error(f"Failed to write pose CSV: {exc}")

    def _notify_ble(self, message: Encode_as_bytes) -> None:
        try:
            asyncio.run_coroutine_threadsafe(self.ble_queue.put(message), self.loop)
        except RuntimeError:
            pass
