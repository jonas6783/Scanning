"""
config/settings.py
Zentrale Konfiguration – alle Parameter an einem Ort.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class CameraConfig:
    depth_width: int   = 1280
    depth_height: int  = 720
    color_width: int   = 1280
    color_height: int  = 720
    fps: int           = 30
    depth_min_m: float = 0.10
    depth_max_m: float = 0.60
    laser_power: int   = 150
    depth_exposure: int = 0
    emitter_enabled: int = 1


@dataclass
class FilterConfig:
    decimation_magnitude: int  = 1
    spatial_magnitude: int     = 2
    spatial_alpha: float       = 0.5
    spatial_delta: int         = 20
    spatial_holes_fill: int    = 0
    temporal_alpha: float      = 0.4
    temporal_delta: int        = 20
    temporal_persistence: int  = 3
    hole_filling_mode: int     = 1
    threshold_min_m: float     = 0.10
    threshold_max_m: float     = 0.60


@dataclass
class CaptureConfig:
    warmup_frames: int       = 60
    average_frames: int      = 15
    turntable_positions: int = 12
    turntable_delay_s: float = 2.0
    min_points_per_frame: int = 50_000

    # rembg Hintergrund-Segmentierung (True = an, False = aus)
    use_rembg: bool = True


@dataclass
class ProcessingConfig:
    voxel_size_m: float            = 0.0005
    sor_nb_neighbors: int          = 20
    sor_std_ratio: float           = 1.0
    icp_max_correspondence_m: float = 0.005
    icp_max_iteration: int         = 100
    icp_relative_fitness: float    = 1e-6
    icp_relative_rmse: float       = 1e-6
    use_global_registration: bool  = True
    fpfh_radius_m: float           = 0.005
    ransac_max_iteration: int      = 100_000
    ransac_confidence: float       = 0.999
    poisson_depth: int             = 10
    poisson_width: int             = 0
    poisson_scale: float           = 1.1
    poisson_linear_fit: bool       = False
    mesh_density_quantile: float   = 0.01
    # Bodenerkennung
    ground_distance_threshold_m: float = 0.005   # 5 mm Toleranz für Ebene
    ground_ransac_n: int               = 3
    ground_ransac_iterations: int      = 1000


# ─────────────────────────────────────────────
# Toleranz-Zonen
# Jede Zone ist ein Quader im Bauteil-Koordinatensystem (in Metern).
# x_min/max, y_min/max, z_min/max relativ zum Bauteil-Zentrum.
# Punkte die in keine Zone fallen bekommen die Default-Toleranz.
# ─────────────────────────────────────────────
@dataclass
class ToleranceZone:
    name: str                    # z.B. "Funktionsfläche oben"
    tolerance_ok_m: float        # grün
    tolerance_warn_m: float      # gelb
    # Bounding Box relativ zum Bauteil-Zentrum (Meter)
    # None = unbegrenzt in dieser Richtung
    x_min: Optional[float] = None
    x_max: Optional[float] = None
    y_min: Optional[float] = None
    y_max: Optional[float] = None
    z_min: Optional[float] = None
    z_max: Optional[float] = None
    color_ok: List[float]   = field(default_factory=lambda: [0.0, 0.8, 0.2])
    color_warn: List[float] = field(default_factory=lambda: [1.0, 0.6, 0.0])
    color_fail: List[float] = field(default_factory=lambda: [0.9, 0.1, 0.1])


@dataclass
class AnalysisConfig:
    # Default-Toleranz (gilt wo keine Zone definiert ist)
    tolerance_ok_m: float       = 0.0005   # 0.5 mm
    tolerance_warn_m: float     = 0.0010   # 1.0 mm
    heatmap_max_m: float        = 0.002
    analysis_sample_points: int = 200_000
    generate_html_report: bool  = True
    report_title: str           = "D405 Qualitätskontrolle – Gussbauteil"

    # Toleranz-Zonen (leer = nur Default-Toleranz)
    # Beispiel-Zonen – anpassen an dein Bauteil!
    zones: List[ToleranceZone] = field(default_factory=lambda: [
        # Funktionsfläche oben – strenge Toleranz
        ToleranceZone(
            name="Funktionsfläche oben",
            tolerance_ok_m=0.0003,    # 0.3 mm
            tolerance_warn_m=0.0006,  # 0.6 mm
            z_min=0.03,               # obere 30 mm des Bauteils
            color_ok=[0.0, 0.9, 0.3],
            color_warn=[1.0, 0.8, 0.0],
            color_fail=[1.0, 0.0, 0.0],
        ),
        # Seitenflächen – mittlere Toleranz
        ToleranceZone(
            name="Seitenflächen",
            tolerance_ok_m=0.0008,    # 0.8 mm
            tolerance_warn_m=0.0015,  # 1.5 mm
            z_min=-0.03,
            z_max=0.03,
            color_ok=[0.0, 0.7, 0.5],
            color_warn=[0.9, 0.5, 0.0],
            color_fail=[0.8, 0.1, 0.1],
        ),
        # Bodenbereich – lockere Toleranz
        ToleranceZone(
            name="Bodenbereich",
            tolerance_ok_m=0.0015,    # 1.5 mm
            tolerance_warn_m=0.003,   # 3.0 mm
            z_max=-0.03,
            color_ok=[0.0, 0.6, 0.8],
            color_warn=[0.8, 0.4, 0.0],
            color_fail=[0.7, 0.1, 0.1],
        ),
    ])


@dataclass
class PathConfig:
    output_dir:  Path = Path("output")
    scans_dir:   Path = Path("output/scans")
    meshes_dir:  Path = Path("output/meshes")
    reports_dir: Path = Path("output/reports")

    def create_all(self):
        for p in [self.output_dir, self.scans_dir, self.meshes_dir, self.reports_dir]:
            p.mkdir(parents=True, exist_ok=True)


@dataclass
class ScannerConfig:
    camera:     CameraConfig     = field(default_factory=CameraConfig)
    filters:    FilterConfig     = field(default_factory=FilterConfig)
    capture:    CaptureConfig    = field(default_factory=CaptureConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    analysis:   AnalysisConfig   = field(default_factory=AnalysisConfig)
    paths:      PathConfig       = field(default_factory=PathConfig)


DEFAULT_CONFIG = ScannerConfig()
