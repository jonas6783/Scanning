"""
core/capture.py
Aufnahme-Logik: Multi-Frame-Mittelung, Qualitätsprüfung, Drehteller-Steuerung.
Konvertiert RealSense-Frames in Open3D-Punktwolken.
"""

import time
from pathlib import Path
from typing import Optional, List
import numpy as np

from config.settings import CaptureConfig, CameraConfig

# rembg – optional, wird nur genutzt wenn installiert
try:
    from rembg import remove as rembg_remove
    from PIL import Image as PILImage
    import io
    REMBG_AVAILABLE = True
except ImportError:
    REMBG_AVAILABLE = False
from core.camera import D405Camera, RS_AVAILABLE
from utils.logger import setup_logger

log = setup_logger("capture")

try:
    import open3d as o3d
    O3D_AVAILABLE = True
except ImportError:
    O3D_AVAILABLE = False
    log.warning("open3d nicht gefunden")

try:
    import pyrealsense2 as rs
except ImportError:
    pass


def _apply_rembg_mask(color_rgb: np.ndarray, depth_array: np.ndarray) -> np.ndarray:
    """
    Nutzt rembg um das Bauteil vom Hintergrund zu trennen.
    Gibt das Tiefenbild zurück, bei dem Hintergrund-Pixel auf 0 gesetzt sind.
    color_rgb: HxWx3 uint8 RGB
    depth_array: HxW uint16
    """
    if not REMBG_AVAILABLE:
        return depth_array

    try:
        import io
        from PIL import Image as PILImage

        # Farbbild → PIL → rembg → Maske extrahieren
        pil_img  = PILImage.fromarray(color_rgb)
        result   = rembg_remove(pil_img)          # RGBA Ausgabe
        result_arr = np.array(result)              # HxWx4
        alpha    = result_arr[:, :, 3]             # Alpha-Kanal = Maske

        # Maske binarisieren (Alpha > 10 = Vordergrund/Bauteil)
        mask = alpha > 10

        # Hintergrund-Pixel im Tiefenbild auf 0 setzen
        depth_masked = depth_array.copy()
        depth_masked[~mask] = 0

        n_removed = int((~mask).sum())
        n_kept    = int(mask.sum())
        return depth_masked

    except Exception as e:
        log.warning(f"rembg Segmentierung fehlgeschlagen, nutze vollständiges Tiefenbild: {e}")
        return depth_array


class FrameCapture:
    """
    Nimmt Einzel- oder Mehrfach-Frames auf und konvertiert sie
    in Open3D-Punktwolken. Implementiert Multi-Frame-Mittelung
    für reduziertes Rauschen.
    """

    def __init__(self, camera: D405Camera, config: CaptureConfig, cam_config: CameraConfig):
        self.camera = camera
        self.cfg = config
        self.cam_cfg = cam_config

    # ──────────────────────────────────────────
    # Einzelaufnahme
    # ──────────────────────────────────────────

    def capture_single(self, label: str = "scan") -> Optional[object]:
        """
        Nimmt einen einzelnen Scan auf.
        Mittelt mehrere Frames für bessere Qualität.
        Gibt eine gefilterte Open3D-Punktwolke zurück.
        """
        log.info(f"Starte Aufnahme: '{label}' ({self.cfg.average_frames} Frames mitteln)")

        depth_stack = []
        color_frame_last = None

        for i in range(self.cfg.average_frames):
            result = self.camera.get_frames()
            if result is None:
                log.warning(f"Frame {i+1} fehlgeschlagen – übersprungen")
                continue

            depth_frame, color_frame = result
            depth_arr = self._frame_to_array(depth_frame)

            if depth_arr is not None:
                depth_stack.append(depth_arr)
                color_frame_last = color_frame

            if (i + 1) % 5 == 0:
                log.debug(f"  {i+1}/{self.cfg.average_frames} Frames aufgenommen")

        if not depth_stack:
            log.error("Keine validen Frames aufgenommen!")
            return None

        # Frames mitteln (Median ist robuster als Mean gegen Ausreißer)
        log.info(f"Mittle {len(depth_stack)} Frames (Median)...")
        depth_averaged = np.median(np.stack(depth_stack, axis=0), axis=0).astype(np.uint16)

        # Qualitätsprüfung
        valid_pixels = np.count_nonzero(depth_averaged)
        total_pixels = depth_averaged.size
        coverage = valid_pixels / total_pixels * 100
        log.info(f"Tiefenbild-Abdeckung: {coverage:.1f}% ({valid_pixels:,} / {total_pixels:,} Pixel)")

        if valid_pixels < self.cfg.min_points_per_frame:
            log.warning(
                f"Nur {valid_pixels:,} valide Punkte – "
                f"Mindest-Schwelle: {self.cfg.min_points_per_frame:,}. "
                f"Prüfe Abstand, Beleuchtung und Scan-Spray!"
            )

        # Zu Open3D Punktwolke konvertieren
        pcd = self._depth_to_pointcloud(depth_averaged, color_frame_last)

        if pcd is not None:
            n = len(pcd.points)
            log.info(f"Punktwolke erstellt: {n:,} Punkte – Label: '{label}'")

        return pcd

    # ──────────────────────────────────────────
    # Drehteller-Scan (mehrere Positionen)
    # ──────────────────────────────────────────

    def capture_turntable(
        self,
        output_dir: Path,
        turntable_controller=None
    ) -> List[object]:
        """
        Nimmt einen vollständigen 360°-Drehteller-Scan auf.
        
        Args:
            output_dir: Verzeichnis für Zwischen-Scans
            turntable_controller: Optionaler Drehteller-Controller
                                  (None = manuelle Bestätigung)
        
        Returns:
            Liste von Open3D-Punktwolken (eine pro Position)
        """
        positions = self.cfg.turntable_positions
        step_deg = 360.0 / positions
        point_clouds = []

        log.info(f"Starte Drehteller-Scan: {positions} Positionen à {step_deg:.1f}°")
        output_dir.mkdir(parents=True, exist_ok=True)

        for i in range(positions):
            angle = i * step_deg
            log.info(f"Position {i+1}/{positions} – {angle:.1f}°")

            if turntable_controller is not None:
                # Programmatische Steuerung (z.B. Schrittmotor)
                turntable_controller.rotate_to(angle)
                time.sleep(self.cfg.turntable_delay_s)
            else:
                # Manuelle Bestätigung
                input(f"\n  Bauteil auf {angle:.0f}° drehen, dann ENTER drücken...")
                time.sleep(0.5)  # Kurz warten bis Bauteil still steht

            label = f"pos_{i+1:02d}_{angle:.0f}deg"
            pcd = self.capture_single(label=label)

            if pcd is not None and len(pcd.points) >= self.cfg.min_points_per_frame:
                # Zwischenspeichern
                save_path = output_dir / f"{label}.ply"
                if O3D_AVAILABLE:
                    o3d.io.write_point_cloud(str(save_path), pcd)
                    log.info(f"Gespeichert: {save_path}")
                point_clouds.append(pcd)
            else:
                log.warning(f"Position {i+1} übersprungen – zu wenige Punkte")

        log.info(f"Drehteller-Scan abgeschlossen: {len(point_clouds)}/{positions} valide Scans")
        return point_clouds

    # ──────────────────────────────────────────
    # Privat: Konvertierung
    # ──────────────────────────────────────────

    def _frame_to_array(self, depth_frame) -> Optional[np.ndarray]:
        """Konvertiert einen RealSense Depth-Frame in ein NumPy-Array."""
        try:
            if RS_AVAILABLE and hasattr(depth_frame, 'get_data'):
                import pyrealsense2 as rs
                data = np.asanyarray(depth_frame.get_data())
            else:
                # Demo-Modus: Fake-Daten
                data = depth_frame.data if hasattr(depth_frame, 'data') else None

            if data is None or data.size == 0:
                return None

            return data.copy()

        except Exception as e:
            log.debug(f"Frame-Konvertierung fehlgeschlagen: {e}")
            return None

    def _depth_to_pointcloud(
        self,
        depth_array: np.ndarray,
        color_frame=None
    ) -> Optional[object]:
        """
        Konvertiert ein Tiefenbild in eine farbige Open3D-Punktwolke.
        Verwendet Kamera-Intrinsics für korrekte 3D-Koordinaten.
        WICHTIG: RealSense depth_scale ist ein Multiplikator (z.B. 0.0001),
        Open3D erwartet einen Divisor → Kehrwert nötig.
        """
        if not O3D_AVAILABLE:
            log.error("open3d nicht installiert! → pip install open3d")
            return None

        try:
            intrinsics = self.camera.get_intrinsics()
            h, w = depth_array.shape

            # Kamera-Matrix aufbauen
            if intrinsics and RS_AVAILABLE:
                fx = intrinsics.fx
                fy = intrinsics.fy
                cx = intrinsics.ppx
                cy = intrinsics.ppy
            else:
                fx = fy = 900.0
                cx, cy = w / 2, h / 2

            # Tiefenskala vom Sensor holen (RealSense: Multiplikator, z.B. 0.0001)
            depth_scale_rs = 0.001  # Fallback
            try:
                if RS_AVAILABLE and self.camera._profile:
                    sensor = self.camera._profile.get_device().first_depth_sensor()
                    depth_scale_rs = sensor.get_depth_scale()
            except Exception as e:
                log.warning(f"Depth-Scale Fallback 0.001 ({e})")

            # Open3D erwartet Divisor: raw / depth_scale_o3d = Meter
            # RealSense liefert Multiplikator: raw * depth_scale_rs = Meter
            # → depth_scale_o3d = 1.0 / depth_scale_rs
            depth_scale_o3d = 1.0 / depth_scale_rs

            nonzero = int((depth_array > 0).sum())
            max_m = float(depth_array.max()) * depth_scale_rs
            depth_trunc = self.cam_cfg.depth_max_m + 0.1  # Puffer gegen Grenzwert

            log.info(
                f"Konvertierung: {w}x{h} | rs_scale={depth_scale_rs:.5f} | "
                f"o3d_scale={depth_scale_o3d:.1f} | nonzero={nonzero:,} | "
                f"max_raw={int(depth_array.max())} | max_m={max_m:.3f}m | "
                f"depth_trunc={depth_trunc:.2f}m"
            )

            if nonzero == 0:
                log.error("Tiefenbild vollständig leer!")
                return None

            # Farbbild holen
            has_color = False
            color_arr = None
            if color_frame is not None:
                try:
                    if RS_AVAILABLE and hasattr(color_frame, 'get_data'):
                        color_arr = np.asanyarray(color_frame.get_data())
                    else:
                        color_arr = color_frame.data
                    color_arr = color_arr[:, :, ::-1].copy()  # BGR → RGB
                    has_color = True
                except Exception as ce:
                    log.warning(f"Farbbild-Konvertierung fehlgeschlagen: {ce}")

            # rembg Segmentierung – Bauteil vom Hintergrund trennen
            if has_color and color_arr is not None and self.cfg.use_rembg:
                depth_array = _apply_rembg_mask(color_arr, depth_array)
                nonzero_after = int((depth_array > 0).sum())
                log.info(f"rembg Maske: {nonzero_after:,} Punkte nach Segmentierung")

            # Open3D Tiefenbild
            depth_o3d = o3d.geometry.Image(depth_array)

            if has_color and color_arr is not None:
                color_o3d = o3d.geometry.Image(color_arr.astype(np.uint8))
            else:
                has_color = False

            # Open3D Intrinsics
            cam_intrinsics = o3d.camera.PinholeCameraIntrinsic(w, h, fx, fy, cx, cy)

            if has_color:
                rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                    color_o3d, depth_o3d,
                    depth_scale=depth_scale_o3d,
                    depth_trunc=depth_trunc,
                    convert_rgb_to_intensity=False
                )
                pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, cam_intrinsics)
            else:
                pcd = o3d.geometry.PointCloud.create_from_depth_image(
                    depth_o3d, cam_intrinsics,
                    depth_scale=depth_scale_o3d,
                    depth_trunc=depth_trunc
                )

            n_pts = len(pcd.points)
            log.info(f"Open3D Punktwolke: {n_pts:,} Punkte")

            if n_pts == 0:
                log.error("Punktwolke leer nach Konvertierung!")
                return None

            # Z-Achse flippen (Open3D-Konvention)
            pcd.transform([[1, 0, 0, 0],
                           [0,-1, 0, 0],
                           [0, 0,-1, 0],
                           [0, 0, 0, 1]])
            return pcd

        except Exception as e:
            import traceback
            log.error(f"Punktwolken-Konvertierung fehlgeschlagen: {e}")
            log.error(traceback.format_exc())
            return None
