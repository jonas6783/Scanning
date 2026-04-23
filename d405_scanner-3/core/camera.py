"""
core/camera.py
D405-Kamera-Initialisierung mit optimierten Parametern für Gussbauteile.
Verwaltet den Pipeline-Lebenszyklus und alle Hardware-Einstellungen.
"""

import time
from typing import Optional, Tuple
import numpy as np

from config.settings import CameraConfig, FilterConfig
from utils.logger import setup_logger

log = setup_logger("camera")

# pyrealsense2 ist optional – bei fehlendem Gerät läuft der Rest im Demo-Modus
try:
    import pyrealsense2 as rs
    RS_AVAILABLE = True
except ImportError:
    RS_AVAILABLE = False
    log.warning("pyrealsense2 nicht gefunden – Demo-Modus aktiv")


class D405Camera:
    """
    Verwaltet den Intel RealSense D405.
    Konfiguriert automatisch alle Parameter für maximale Qualität
    bei Gussbauteil-Scans im Nahbereich (15–25 cm).
    """

    def __init__(self, config: CameraConfig, filter_config: FilterConfig):
        self.cfg = config
        self.fcfg = filter_config
        self._pipeline: Optional[object] = None
        self._profile: Optional[object] = None
        self._align: Optional[object] = None
        self._filters: list = []
        self._intrinsics: Optional[object] = None
        self.running = False

    # ──────────────────────────────────────────
    # Starten / Stoppen
    # ──────────────────────────────────────────

    def start(self) -> bool:
        """Startet die Kamera-Pipeline mit optimierten Einstellungen."""
        if not RS_AVAILABLE:
            log.warning("Kamera-Start übersprungen – Demo-Modus")
            self.running = True
            return True

        try:
            self._pipeline = rs.pipeline()
            rs_config = rs.config()

            # Streams konfigurieren
            rs_config.enable_stream(
                rs.stream.depth,
                self.cfg.depth_width, self.cfg.depth_height,
                rs.format.z16, self.cfg.fps
            )
            rs_config.enable_stream(
                rs.stream.color,
                self.cfg.color_width, self.cfg.color_height,
                rs.format.bgr8, self.cfg.fps
            )

            log.info(f"Starte Pipeline: {self.cfg.depth_width}×{self.cfg.depth_height} @ {self.cfg.fps} fps")
            self._profile = self._pipeline.start(rs_config)

            # Depth-Sensor-Einstellungen
            self._configure_depth_sensor()

            # Depth → Color Ausrichten
            self._align = rs.align(rs.stream.color)

            # Post-Processing Filter aufbauen
            self._build_filters()

            # Intrinsics speichern
            depth_stream = self._profile.get_stream(rs.stream.depth)
            self._intrinsics = depth_stream.as_video_stream_profile().get_intrinsics()

            log.info(
                f"Intrinsics: fx={self._intrinsics.fx:.1f} fy={self._intrinsics.fy:.1f} "
                f"ppx={self._intrinsics.ppx:.1f} ppy={self._intrinsics.ppy:.1f}"
            )

            self.running = True
            log.info("D405 erfolgreich gestartet")
            return True

        except Exception as e:
            log.error(f"Kamera-Start fehlgeschlagen: {e}")
            return False

    def stop(self):
        """Stoppt die Pipeline sauber."""
        if self._pipeline and self.running:
            self._pipeline.stop()
        self.running = False
        log.info("D405 Pipeline gestoppt")

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_):
        self.stop()

    # ──────────────────────────────────────────
    # Frame-Aufnahme
    # ──────────────────────────────────────────

    def get_frames(self, timeout_ms: int = 5000) -> Optional[Tuple]:
        """
        Gibt ein (depth_frame, color_frame) Paar zurück.
        Wendet alle konfigurierten Post-Processing Filter an.
        Gibt None zurück bei Timeout oder Fehler.
        """
        if not RS_AVAILABLE or not self.running:
            return self._demo_frames()

        try:
            frameset = self._pipeline.wait_for_frames(timeout_ms=timeout_ms)

            # Depth auf Color ausrichten
            aligned = self._align.process(frameset)
            depth_frame = aligned.get_depth_frame()
            color_frame = aligned.get_color_frame()

            if not depth_frame or not color_frame:
                log.debug("Leerer Frame empfangen – übersprungen")
                return None

            # Post-Processing anwenden
            for f in self._filters:
                depth_frame = f.process(depth_frame)

            return depth_frame, color_frame

        except Exception as e:
            log.error(f"Frame-Aufnahme fehlgeschlagen: {e}")
            return None

    def get_intrinsics(self) -> Optional[object]:
        """Gibt die Kamera-Intrinsics zurück."""
        return self._intrinsics

    def warmup(self, frames: int = 60):
        """
        Lässt die Kamera einpendeln.
        Wichtig: Auto-Exposure und Laser brauchen ~2 Sekunden.
        """
        log.info(f"Kamera einpendeln lassen ({frames} Frames)...")
        discarded = 0
        for _ in range(frames):
            result = self.get_frames()
            if result:
                discarded += 1
        log.info(f"Warmup abgeschlossen ({discarded} Frames verworfen)")

    # ──────────────────────────────────────────
    # Privat: Konfiguration
    # ──────────────────────────────────────────

    def _configure_depth_sensor(self):
        """Setzt alle Depth-Sensor-Parameter für optimale Qualität."""
        if not RS_AVAILABLE:
            return

        device = self._profile.get_device()
        depth_sensor = device.first_depth_sensor()

        options = {
            rs.option.laser_power:       self.cfg.laser_power,
            rs.option.emitter_enabled:   self.cfg.emitter_enabled,
            rs.option.min_distance:      int(self.cfg.depth_min_m * 1000),
        }

        if self.cfg.depth_exposure > 0:
            options[rs.option.exposure] = self.cfg.depth_exposure
        else:
            # Auto-Exposure aktivieren
            try:
                depth_sensor.set_option(rs.option.enable_auto_exposure, 1)
                log.debug("Auto-Exposure aktiviert")
            except Exception:
                pass

        for option, value in options.items():
            try:
                if depth_sensor.supports(option):
                    depth_sensor.set_option(option, value)
                    log.debug(f"Gesetzt: {option} = {value}")
            except Exception as e:
                log.debug(f"Option nicht unterstützt: {option} ({e})")

        log.info(
            f"Sensor konfiguriert: Laser={self.cfg.laser_power}, "
            f"Bereich={self.cfg.depth_min_m*100:.0f}–{self.cfg.depth_max_m*100:.0f} cm"
        )

    def _build_filters(self):
        """Baut die Post-Processing Filter-Pipeline auf."""
        if not RS_AVAILABLE:
            return

        self._filters = []
        f = self.fcfg

        # 1. Decimation (optional – für schnellere Verarbeitung)
        if f.decimation_magnitude > 1:
            dec = rs.decimation_filter()
            dec.set_option(rs.option.filter_magnitude, f.decimation_magnitude)
            self._filters.append(dec)
            log.debug(f"Decimation Filter: magnitude={f.decimation_magnitude}")

        # 2. Threshold Filter (Tiefenbereich begrenzen)
        thresh = rs.threshold_filter()
        thresh.set_option(rs.option.min_distance, f.threshold_min_m)
        thresh.set_option(rs.option.max_distance, f.threshold_max_m)
        self._filters.append(thresh)

        # 3. Spatial Filter (räumliche Glättung, Kanten erhalten)
        spatial = rs.spatial_filter()
        spatial.set_option(rs.option.filter_magnitude,   f.spatial_magnitude)
        spatial.set_option(rs.option.filter_smooth_alpha, f.spatial_alpha)
        spatial.set_option(rs.option.filter_smooth_delta, f.spatial_delta)
        spatial.set_option(rs.option.holes_fill,         f.spatial_holes_fill)
        self._filters.append(spatial)

        # 4. Temporal Filter (zeitliche Mittelung – Rauschen)
        temporal = rs.temporal_filter()
        temporal.set_option(rs.option.filter_smooth_alpha, f.temporal_alpha)
        temporal.set_option(rs.option.filter_smooth_delta, f.temporal_delta)
        temporal.set_option(rs.option.holes_fill,         f.temporal_persistence)
        self._filters.append(temporal)

        # 5. Hole Filling (konservativ)
        hole_fill = rs.hole_filling_filter()
        hole_fill.set_option(rs.option.holes_fill, f.hole_filling_mode)
        self._filters.append(hole_fill)

        log.info(f"Filter-Pipeline aufgebaut: {len(self._filters)} Filter aktiv")

    def _demo_frames(self) -> Tuple:
        """Generiert synthetische Demo-Frames wenn keine Kamera vorhanden."""
        import numpy as np
        w, h = self.cfg.depth_width, self.cfg.depth_height

        class FakeDepthFrame:
            def __init__(self):
                self.data = np.random.randint(150, 350, (h, w), dtype=np.uint16)
            def get_data(self): return self.data
            def get_width(self): return w
            def get_height(self): return h

        class FakeColorFrame:
            def __init__(self):
                self.data = np.random.randint(80, 180, (h, w, 3), dtype=np.uint8)
            def get_data(self): return self.data

        return FakeDepthFrame(), FakeColorFrame()
