"""
config/settings.py
Zentrale Konfiguration – alle Parameter an einem Ort.
Anpassen je nach Bauteil, Beleuchtung und gewünschter Qualität.
"""

from dataclasses import dataclass, field
from pathlib import Path


# ─────────────────────────────────────────────
# Kamera-Parameter (D405 optimiert für 15–25 cm)
# ─────────────────────────────────────────────
@dataclass
class CameraConfig:
    # Auflösung & Framerate
    depth_width: int  = 1280
    depth_height: int = 720
    color_width: int  = 1280
    color_height: int = 720
    fps: int          = 30

    # Tiefenbereich in Metern (D405 optimal 0.07–0.6 m)
    depth_min_m: float = 0.10   # 10 cm – nah herangehen
    depth_max_m: float = 0.50   # 50 cm – Sicherheitspuffer

    # Laser (100–200 empfohlen – Überstrahlung vermeiden)
    laser_power: int   = 150

    # Belichtung Depth-Stream (0 = Auto)
    depth_exposure: int = 0

    # Emitter Muster (1 = an, 0 = aus)
    emitter_enabled: int = 1


# ─────────────────────────────────────────────
# Post-Processing Filter (RealSense SDK)
# ─────────────────────────────────────────────
@dataclass
class FilterConfig:
    # Decimation – Auflösungsreduktion (1 = keine)
    decimation_magnitude: int = 1   # Bei großen Bauteilen auf 2 erhöhen

    # Spatial Filter – räumliche Glättung (Kanten erhalten)
    spatial_magnitude: int   = 2
    spatial_alpha: float     = 0.5
    spatial_delta: int       = 20
    spatial_holes_fill: int  = 0

    # Temporal Filter – zeitliche Mittelung (Rauschen reduzieren)
    temporal_alpha: float    = 0.4
    temporal_delta: int      = 20
    temporal_persistence: int = 3   # RS2_HOLES_FILL_PERSISTENT

    # Hole Filling (1 = farthest, am konservativsten)
    hole_filling_mode: int   = 1

    # Threshold Filter
    threshold_min_m: float   = 0.10
    threshold_max_m: float   = 0.50


# ─────────────────────────────────────────────
# Aufnahme-Parameter
# ─────────────────────────────────────────────
@dataclass
class CaptureConfig:
    # Anzahl Frames vor der eigentlichen Aufnahme (Kamera einpendeln lassen)
    warmup_frames: int      = 60

    # Frames die gemittelt werden (mehr = besser, aber langsamer)
    average_frames: int     = 15

    # Drehteller-Scan: Anzahl Positionen für 360°
    turntable_positions: int = 12   # 30° Schritte

    # Wartezeit nach Drehteller-Schritt (Sekunden)
    turntable_delay_s: float = 2.0

    # Mindest-Punktanzahl pro Frame (Qualitätskontrolle)
    min_points_per_frame: int = 50_000


# ─────────────────────────────────────────────
# Punktwolken-Verarbeitung
# ─────────────────────────────────────────────
@dataclass
class ProcessingConfig:
    # Voxel-Downsampling Größe in Metern (None = kein Downsampling)
    voxel_size_m: float = 0.0005   # 0.5 mm – guter Kompromiss

    # SOR Outlier Removal
    sor_nb_neighbors: int    = 20
    sor_std_ratio: float     = 1.0   # Strenger = mehr Rauschen entfernt

    # ICP Registrierung
    icp_max_correspondence_m: float = 0.005  # 5 mm – Starttoleranz
    icp_max_iteration: int          = 100
    icp_relative_fitness: float     = 1e-6
    icp_relative_rmse: float        = 1e-6

    # Globale Registrierung (FPFH + RANSAC) vor ICP
    use_global_registration: bool   = True
    fpfh_radius_m: float            = 0.005
    ransac_max_iteration: int       = 100_000
    ransac_confidence: float        = 0.999

    # Poisson Mesh-Rekonstruktion
    poisson_depth: int              = 10   # 10–12 für Guss empfohlen
    poisson_width: int              = 0
    poisson_scale: float            = 1.1
    poisson_linear_fit: bool        = False

    # Mesh-Dichte Threshold (entfernt dünn besetzte Oberflächen)
    mesh_density_quantile: float    = 0.01  # unterste 1% entfernen


# ─────────────────────────────────────────────
# Qualitätsanalyse / CAD-Vergleich
# ─────────────────────────────────────────────
@dataclass
class AnalysisConfig:
    # Maximale akzeptable Abweichung in Metern
    tolerance_ok_m: float      = 0.0005  # 0.5 mm   – grün
    tolerance_warn_m: float    = 0.0010  # 1.0 mm   – gelb
    # > tolerance_warn_m         = rot

    # Heatmap Farbskala (symmetrisch um 0)
    heatmap_max_m: float       = 0.002   # ±2 mm Skala

    # Anzahl Punkte für Stichprobenanalyse
    analysis_sample_points: int = 200_000

    # Report
    generate_html_report: bool  = True
    report_title: str           = "D405 Qualitätskontrolle – Gussbauteil"


# ─────────────────────────────────────────────
# Pfade
# ─────────────────────────────────────────────
@dataclass
class PathConfig:
    output_dir: Path = Path("output")
    scans_dir:  Path = Path("output/scans")
    meshes_dir: Path = Path("output/meshes")
    reports_dir:Path = Path("output/reports")

    def create_all(self):
        for p in [self.output_dir, self.scans_dir, self.meshes_dir, self.reports_dir]:
            p.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────
# Master-Konfiguration
# ─────────────────────────────────────────────
@dataclass
class ScannerConfig:
    camera:     CameraConfig     = field(default_factory=CameraConfig)
    filters:    FilterConfig     = field(default_factory=FilterConfig)
    capture:    CaptureConfig    = field(default_factory=CaptureConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    analysis:   AnalysisConfig   = field(default_factory=AnalysisConfig)
    paths:      PathConfig       = field(default_factory=PathConfig)


# Globale Standard-Instanz
DEFAULT_CONFIG = ScannerConfig()
