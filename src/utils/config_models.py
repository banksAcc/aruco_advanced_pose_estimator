"""Typed representations of the YAML application configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Mapping, MutableMapping, Optional,  Literal, TypeAlias, Union
import asyncio
from datetime import datetime
import numpy as np

def _as_mapping(raw: Any) -> Mapping[str, Any]:
    """Return a mapping view for ``raw`` falling back to an empty dict."""

    if isinstance(raw, Mapping):
        return raw
    return {}


def _coerce_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "y", "on"}:
            return True
        if lowered in {"false", "0", "no", "n", "off"}:
            return False
    if isinstance(value, (int, float)):
        return bool(value)
    return default


def _coerce_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class BleConfig:
    """Configuration for BLE discovery and connection."""
    
    message_at_computation_start: str
    message_at_computation_end: str
    message_at_grab_start: str
    message_at_grab_end: str
    nus_tx_uuid: str
    nus_rx_uuid: str
    name_prefix: str = "ESP32-RGB-BLE"
    addr: Optional[str] = None
    scan_timeout: float = 8.0

    @classmethod
    def from_mapping(cls, raw: Any) -> "BleConfig":
        data = _as_mapping(raw)
        return cls(
            message_at_computation_start=str(data.get("message_at_computation_start", "")),
            message_at_computation_end=str(data.get("message_at_computation_end", "")),
            message_at_grab_start=str(data.get("message_at_grab_start", "")),
            message_at_grab_end=str(data.get("message_at_grab_end", "")),
            nus_tx_uuid=str(data.get("nus_tx_uuid", "")),
            nus_rx_uuid=str(data.get("nus_rx_uuid", "")),
            name_prefix=str(data.get("name_prefix", cls.name_prefix)),
            addr=data.get("addr") or None,
            scan_timeout=_coerce_float(data.get("scan_timeout"), cls.scan_timeout),
        )


@dataclass(frozen=True)
class CaptureConfig:
    """Configuration for frame acquisition and persistence."""

    frequency_ms: int = 200
    output_root: Path = Path("../captures")
    save_frames: bool = False
    frame_queue_size: int = 4
    simulate_camera: bool = False
    camera_type: str = "pylon"
    camera_id: int = 0
    camera_serial: Optional[str] = None
    camera_ip: Optional[str] = None
    keep_camera_warm: bool = True
    test_source_dir: Path = Path("../image_to_be_used")

    @classmethod
    def from_mapping(cls, raw: Any) -> "CaptureConfig":
        data = _as_mapping(raw)
        output_root = Path(data.get("output_root", cls.output_root))
        test_source_dir = Path(data.get("test_source_dir", cls.test_source_dir))
        camera_serial = data.get("camera_serial")
        camera_ip = data.get("camera_ip")
        return cls(
            frequency_ms=_coerce_int(data.get("frequency_ms"), cls.frequency_ms),
            output_root=output_root,
            save_frames=_coerce_bool(data.get("save_frames"), cls.save_frames),
            frame_queue_size=_coerce_int(data.get("frame_queue_size"), cls.frame_queue_size),
            simulate_camera=_coerce_bool(data.get("simulate_camera"), cls.simulate_camera),
            camera_type=str(data.get("camera_type", cls.camera_type)),
            camera_id=_coerce_int(data.get("camera_id"), cls.camera_id),
            camera_serial=str(camera_serial) if camera_serial is not None else None,
            camera_ip=str(camera_ip) if camera_ip is not None else None,
            keep_camera_warm=_coerce_bool(data.get("keep_camera_warm"), cls.keep_camera_warm),
            test_source_dir=test_source_dir,
        )

@dataclass(frozen=True)
class IcoPoseConfig:
    """Configurazione specifica per il backend icosaedro troncato."""
    dictionary: str = "4X4_50"
    marker_size_mm: float = 21.33
    transform_file: Optional[Path] = None
    geo_radius: float = 0.061035
    z_shift_tip: float = 0.20620

    @classmethod
    def from_mapping(cls, raw: Any) -> "IcoPoseConfig":
        data = _as_mapping(raw)
        tf = data.get("transform_file")
        return cls(
            dictionary=str(data.get("dictionary", cls.dictionary)),
            marker_size_mm=_coerce_float(data.get("marker_size_mm"), cls.marker_size_mm),
            transform_file=Path(tf) if tf else None,
            geo_radius=_coerce_float(data.get("geo_radius"), cls.geo_radius),
            z_shift_tip=_coerce_float(data.get("z_shift_tip"), cls.z_shift_tip),
        )


@dataclass(frozen=True)
class MarkerFilterConfig:
    """Configurazione per il filtraggio e pesatura dei marker (marker_filter_average)."""
    min_area: float = 100.0
    weight_exponent: float = 1.5
    outliers_threshold: float = 0.030

    @classmethod
    def from_mapping(cls, raw: Any) -> "MarkerFilterConfig":
        data = _as_mapping(raw)
        return cls(
            min_area=_coerce_float(data.get("min_area"), cls.min_area),
            weight_exponent=_coerce_float(data.get("weight_exponent"), cls.weight_exponent),
            outliers_threshold=_coerce_float(data.get("outliers_threshold"), cls.outliers_threshold),
        )

@dataclass(frozen=True)
class SubpixelConfig:
    """Parametri per l'affinamento subpixel (subpixel_config)."""
    subpixel_iter: int = 500
    subpixel_eps: float = 0.0001

    @classmethod
    def from_mapping(cls, raw: Any) -> "SubpixelConfig":
        data = _as_mapping(raw)
        # Gestiamo il typo 'subpixe_iter' presente nel tuo YAML
        return cls(
            subpixel_iter=_coerce_int(data.get("subpixel_iter"), cls.subpixel_iter),
            subpixel_eps=_coerce_float(data.get("subpixel_eps"), cls.subpixel_eps),
        )

@dataclass(frozen=True)
class ExtrinsicConfig:
    trans_x_mm: float = 0.0
    trans_y_mm: float = 0.0
    trans_z_mm: float = 0.0
    rot_phi_deg: float = 0.0
    rot_theta_deg: float = 0.0
    rot_psi_deg: float = 0.0

    @classmethod
    def from_mapping(cls, raw: Any) -> "ExtrinsicConfig":
        data = _as_mapping(raw)
        return cls(
            trans_x_mm=float(data.get("trans_x_mm", 0.0)),
            trans_y_mm=float(data.get("trans_y_mm", 0.0)),
            trans_z_mm=float(data.get("trans_z_mm", 0.0)),
            rot_phi_deg=float(data.get("rot_phi_deg", 0.0)),
            rot_theta_deg=float(data.get("rot_theta_deg", 0.0)),
            rot_psi_deg=float(data.get("rot_psi_deg", 0.0)),
        )

@dataclass(frozen=True)
class PoseConfig:
    """Configurazione generale per il sistema di posa."""
    enabled: bool = True
    method: str = "ico"
    max_parallel_jobs: int = 5
    save_overlay: bool = True
    camera_calibration_npz: Optional[Path] = None
    extrinsic_calibration: ExtrinsicConfig = field(default_factory=ExtrinsicConfig)
    ico: IcoPoseConfig = field(default_factory=IcoPoseConfig)
    marker_filter_average: MarkerFilterConfig = field(default_factory=MarkerFilterConfig)
    subpixel_config: SubpixelConfig = field(default_factory=SubpixelConfig)

    @classmethod
    def from_mapping(cls, raw: Any) -> "PoseConfig":
        data = _as_mapping(raw)
        calib = data.get("camera_calibration_npz")
        return cls(
            enabled=_coerce_bool(data.get("enabled"), cls.enabled),
            method=str(data.get("method", cls.method)),
            max_parallel_jobs=_coerce_int(data.get("max_parallel_jobs"), cls.max_parallel_jobs),
            save_overlay=_coerce_bool(data.get("save_overlay"), cls.save_overlay),
            camera_calibration_npz=Path(calib) if calib else None,
            extrinsic_calibration=ExtrinsicConfig.from_mapping(data.get("extrinsic_calibration", {})),
            ico=IcoPoseConfig.from_mapping(data.get("ico", {})),
            marker_filter_average=MarkerFilterConfig.from_mapping(data.get("marker_filter_average", {})),
            subpixel_config=SubpixelConfig.from_mapping(data.get("subpixel_config", {})),
        )


@dataclass(frozen=True)
class RuntimeConfig:
    """Generic runtime flags that influence logging and debugging."""

    debug: bool = False
    log_level: str = "INFO"
    log_to_file: bool = False

    @classmethod
    def from_mapping(cls, raw: Any) -> "RuntimeConfig":
        data = _as_mapping(raw)
        return cls(
            debug=_coerce_bool(data.get("debug"), cls.debug),
            log_level=str(data.get("log_level", cls.log_level)),
            log_to_file=_coerce_bool(data.get("log_to_file"), cls.log_to_file),
        )


@dataclass(frozen=True)
class PlotConfig:
    """Configuration for optional plotting utilities."""

    default_pose_json: Optional[Path] = None

    @classmethod
    def from_mapping(cls, raw: Any) -> "PlotConfig":
        data = _as_mapping(raw)
        default_json = data.get("default_pose_json")
        return cls(default_pose_json=Path(default_json) if default_json else None)

@dataclass(frozen=True)
class IcosahedronConfig:
    edge_length: float
    size_ratio: float
    lift_dist: float
    manual_offsets: Mapping[str, float]

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "IcosahedronConfig":
        return cls(
            edge_length=float(raw.get("edge_length", 0.025)),
            size_ratio=float(raw.get("size_ratio", 0.85)),
            lift_dist=float(raw.get("lift_dist", 0.002)),
            manual_offsets=_as_mapping(raw.get("manual_offsets", {})),
        )
    
@dataclass(frozen=True)
class AppConfig:
    """Aggregated application configuration with strongly-typed sections."""

    ble: BleConfig
    capture: CaptureConfig
    pose: PoseConfig
    runtime: RuntimeConfig
    plot: PlotConfig
    build_icosahedron: IcosahedronConfig
    marker_map: Mapping[int, str] 
    raw: MutableMapping[str, Any] = field(repr=False)

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "AppConfig":
        """Crea la configurazione gestendo la conversione dei tipi dallo YAML."""
        
        # Gestione della marker_map dallo YAML
        raw_map = _as_mapping(raw.get("marker_map", {}))
        # Convertiamo le chiavi in interi per gli ID ArUco
        processed_map = {int(k): str(v) for k, v in raw_map.items()}

        return cls(
            ble=BleConfig.from_mapping(raw.get("ble", {})),
            capture=CaptureConfig.from_mapping(raw.get("capture", {})),
            pose=PoseConfig.from_mapping(raw.get("pose", {})),
            runtime=RuntimeConfig.from_mapping(raw.get("runtime", {})),
            plot=PlotConfig.from_mapping(raw.get("plot", {})),
            build_icosahedron=IcosahedronConfig.from_mapping(raw.get("build_icosahedron", {})),
            marker_map=processed_map,
            raw=dict(raw),
        )

    def as_dict(self) -> dict[str, Any]:
        return dict(self.raw)

@dataclass(frozen=True)
class PoseStartMessage:
    """Payload emitted when a capture session starts."""

    session_key: str
    session_dir: Path
    frame_queue: asyncio.Queue[Optional[FramePacket]]
    start: str
    freq_ms: int
    label: str
    save_frames: bool
    save_dir: Optional[Path] = None
    action: Literal["start"] = "start"

@dataclass(frozen=True)
class PoseEndMessage:
    """Payload emitted once a capture session finishes."""

    session_key: str
    session_dir: Path
    start: str
    end: str
    freq_ms: int
    label: str
    save_dir: Optional[Path] = None
    action: Literal["end"] = "end"


PoseWorkerPayload: TypeAlias = Union[PoseStartMessage, PoseEndMessage]

@dataclass
class FramePacket:
    """Container holding a single captured frame and associated metadata."""

    session_key: str
    index: int
    timestamp: float
    frame: "np.ndarray"
    filename: str
    save_path: Optional[Path] = None

    @property
    def iso_timestamp(self) -> str:
        """Return the capture timestamp encoded as an ISO string."""

        return datetime.fromtimestamp(self.timestamp).isoformat()


@dataclass
class SessionJob:
    """Track state for an in-flight pose estimation session."""
    key: str
    frame_queue: asyncio.Queue[Optional[FramePacket]]
    freq_ms: int
    start_iso: str
    results: dict[str, Any]
    label: str
    save_frames: bool
    save_dir: Optional[Path] = None
    save_overlay: bool = True
    end_iso: Optional[str] = None
    finished: asyncio.Event = field(default_factory=asyncio.Event)
    task: Optional[asyncio.Task] = None
    overlay_paths: List[Path] = field(default_factory=list)