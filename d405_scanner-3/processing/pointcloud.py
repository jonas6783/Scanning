"""
processing/pointcloud.py
Punktwolken-Verarbeitung: Downsampling, Outlier-Entfernung,
globale Registrierung (FPFH+RANSAC) und feines ICP-Alignment.
"""

from typing import List, Optional, Tuple
import numpy as np

from config.settings import ProcessingConfig
from utils.logger import setup_logger

log = setup_logger("pointcloud")

try:
    import open3d as o3d
    O3D_AVAILABLE = True
except ImportError:
    O3D_AVAILABLE = False


class PointCloudProcessor:
    """
    Verarbeitet Rohdaten-Punktwolken zu einer sauberen,
    registrierten Gesamtpunktwolke aus mehreren Einzelscans.
    """

    def __init__(self, config: ProcessingConfig):
        self.cfg = config

    # ──────────────────────────────────────────
    # Vorverarbeitung
    # ──────────────────────────────────────────

    def preprocess(self, pcd: object, label: str = "") -> object:
        """
        Vollständige Vorverarbeitung einer Rohdaten-Punktwolke:
        1. Voxel-Downsampling
        2. Normalen-Schätzung
        3. Statistical Outlier Removal

        Args:
            pcd:   Open3D PointCloud
            label: Name für Log-Ausgaben

        Returns:
            Vorverarbeitete Open3D PointCloud
        """
        if not O3D_AVAILABLE:
            return pcd

        n_start = len(pcd.points)
        log.info(f"Vorverarbeitung '{label}': {n_start:,} Punkte")

        # 1. Voxel-Downsampling (gleichmäßige Punktdichte)
        pcd_down = pcd.voxel_down_sample(voxel_size=self.cfg.voxel_size_m)
        log.debug(f"  Nach Downsampling: {len(pcd_down.points):,} Punkte "
                  f"(Voxel={self.cfg.voxel_size_m*1000:.1f} mm)")

        # 2. Normalen schätzen (notwendig für ICP und Mesh)
        pcd_down.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=self.cfg.voxel_size_m * 2,
                max_nn=30
            )
        )
        pcd_down.orient_normals_consistent_tangent_plane(100)
        log.debug("  Normalen geschätzt und ausgerichtet")

        # 3. Statistical Outlier Removal
        pcd_clean, ind = pcd_down.remove_statistical_outlier(
            nb_neighbors=self.cfg.sor_nb_neighbors,
            std_ratio=self.cfg.sor_std_ratio
        )
        removed = len(pcd_down.points) - len(pcd_clean.points)
        log.info(
            f"  SOR: {removed:,} Ausreißer entfernt → "
            f"{len(pcd_clean.points):,} Punkte verbleiben"
        )

        return pcd_clean

    # ──────────────────────────────────────────
    # Registrierung
    # ──────────────────────────────────────────

    def register_multiple(self, point_clouds: List[object]) -> Optional[object]:
        """
        Registriert mehrere Einzelscans zu einer gemeinsamen Punktwolke.
        Verwendet globale Registrierung (FPFH+RANSAC) gefolgt von feinem ICP.

        Args:
            point_clouds: Liste von Open3D PointClouds (bereits vorverarbeitet)

        Returns:
            Zusammengeführte Gesamt-Punktwolke
        """
        if not O3D_AVAILABLE:
            log.warning("open3d nicht verfügbar")
            return None

        if len(point_clouds) == 0:
            log.error("Keine Punktwolken zur Registrierung")
            return None

        if len(point_clouds) == 1:
            log.info("Nur ein Scan – keine Registrierung nötig")
            return point_clouds[0]

        log.info(f"Registriere {len(point_clouds)} Scans...")

        # Erster Scan ist die Referenz
        combined = point_clouds[0]
        total_registered = 1

        for i, pcd_target in enumerate(point_clouds[1:], start=2):
            log.info(f"Registriere Scan {i}/{len(point_clouds)}...")

            # Beide Wolken für Registrierung vorbereiten
            source_fpfh = self._compute_fpfh(pcd_target)
            target_fpfh = self._compute_fpfh(combined)

            # Schritt 1: Globale Registrierung (grobe Ausrichtung)
            if self.cfg.use_global_registration:
                result_global = self._global_registration(
                    pcd_target, combined, source_fpfh, target_fpfh
                )
                initial_transform = result_global.transformation
                log.debug(
                    f"  Globale Registrierung: fitness={result_global.fitness:.4f}, "
                    f"RMSE={result_global.inlier_rmse*1000:.3f} mm"
                )
            else:
                initial_transform = np.eye(4)

            # Schritt 2: Feines ICP
            result_icp = self._icp_registration(
                pcd_target, combined, initial_transform
            )

            if result_icp.fitness < 0.3:
                log.warning(
                    f"  Scan {i}: Schlechte ICP-Konvergenz "
                    f"(fitness={result_icp.fitness:.3f}) – übersprungen"
                )
                continue

            log.info(
                f"  ICP: fitness={result_icp.fitness:.4f}, "
                f"RMSE={result_icp.inlier_rmse*1000:.3f} mm"
            )

            # Transformierten Scan zur Gesamtwolke hinzufügen
            pcd_aligned = pcd_target.transform(result_icp.transformation)
            combined = self._merge_clouds([combined, pcd_aligned])
            total_registered += 1

        log.info(
            f"Registrierung abgeschlossen: {total_registered}/{len(point_clouds)} Scans "
            f"→ {len(combined.points):,} Gesamtpunkte"
        )

        # Abschließendes Downsampling der Gesamtwolke
        combined_clean = combined.voxel_down_sample(voxel_size=self.cfg.voxel_size_m)
        combined_clean.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=self.cfg.voxel_size_m * 2, max_nn=30
            )
        )
        log.info(f"Finale Punktwolke: {len(combined_clean.points):,} Punkte")

        return combined_clean

    # ──────────────────────────────────────────
    # Privat: Registrierungs-Hilfsmethoden
    # ──────────────────────────────────────────

    def _compute_fpfh(self, pcd: object) -> object:
        """Berechnet FPFH-Features für globale Registrierung."""
        radius = self.cfg.fpfh_radius_m

        # Normalen sicherstellen
        if not pcd.has_normals():
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(
                    radius=radius * 2, max_nn=30
                )
            )

        fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pcd,
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 5, max_nn=100)
        )
        return fpfh

    def _global_registration(
        self,
        source: object,
        target: object,
        source_fpfh: object,
        target_fpfh: object
    ) -> object:
        """RANSAC-basierte globale Registrierung."""
        distance_threshold = self.cfg.voxel_size_m * 1.5

        result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source, target,
            source_fpfh, target_fpfh,
            mutual_filter=True,
            max_correspondence_distance=distance_threshold,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            ransac_n=3,
            checkers=[
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
            ],
            criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(
                self.cfg.ransac_max_iteration,
                self.cfg.ransac_confidence
            )
        )
        return result

    def _icp_registration(
        self,
        source: object,
        target: object,
        initial_transform: np.ndarray
    ) -> object:
        """Feines ICP-Alignment (Point-to-Plane für bessere Konvergenz)."""
        result = o3d.pipelines.registration.registration_icp(
            source, target,
            max_correspondence_distance=self.cfg.icp_max_correspondence_m,
            init=initial_transform,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
                relative_fitness=self.cfg.icp_relative_fitness,
                relative_rmse=self.cfg.icp_relative_rmse,
                max_iteration=self.cfg.icp_max_iteration
            )
        )
        return result

    def _merge_clouds(self, clouds: List[object]) -> object:
        """Fügt mehrere Punktwolken zusammen."""
        merged = o3d.geometry.PointCloud()
        for pcd in clouds:
            merged += pcd
        return merged

    # ──────────────────────────────────────────
    # Statistiken
    # ──────────────────────────────────────────

    def compute_stats(self, pcd: object) -> dict:
        """Berechnet Qualitätsstatistiken der Punktwolke."""
        if not O3D_AVAILABLE or pcd is None:
            return {}

        points = np.asarray(pcd.points)
        if len(points) == 0:
            return {}

        bbox = pcd.get_axis_aligned_bounding_box()
        extent = bbox.get_extent()

        stats = {
            "n_points":    len(points),
            "bbox_x_mm":   extent[0] * 1000,
            "bbox_y_mm":   extent[1] * 1000,
            "bbox_z_mm":   extent[2] * 1000,
            "center_x_mm": np.mean(points[:, 0]) * 1000,
            "center_y_mm": np.mean(points[:, 1]) * 1000,
            "center_z_mm": np.mean(points[:, 2]) * 1000,
            "has_colors":  pcd.has_colors(),
            "has_normals": pcd.has_normals(),
        }

        log.info(
            f"Punktwolken-Stats: {stats['n_points']:,} Punkte, "
            f"Bounding Box: {stats['bbox_x_mm']:.1f} × {stats['bbox_y_mm']:.1f} × "
            f"{stats['bbox_z_mm']:.1f} mm"
        )
        return stats
