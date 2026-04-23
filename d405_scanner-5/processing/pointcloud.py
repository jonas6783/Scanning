"""
processing/pointcloud.py
Punktwolken-Verarbeitung mit GPU-Beschleunigung (Open3D CUDA Tensor API).
Fällt automatisch auf CPU zurück wenn CUDA nicht verfügbar.

GPU-beschleunigt:
  - Voxel Downsampling
  - Normalen-Schätzung
  - ICP Registrierung (Point-to-Plane)
  - FPFH Feature-Berechnung

CPU (kein CUDA-Support in Open3D):
  - SOR Outlier Removal → ersetzt durch Radius Outlier Removal auf GPU
  - Poisson Mesh-Rekonstruktion
"""

from typing import List, Optional
import numpy as np
import time

from config.settings import ProcessingConfig
from utils.logger import setup_logger

log = setup_logger("pointcloud")

try:
    import open3d as o3d
    import open3d.core as o3c
    O3D_AVAILABLE = True
    CUDA_AVAILABLE = o3d.core.cuda.is_available()
    DEVICE = o3c.Device("CUDA:0") if CUDA_AVAILABLE else o3c.Device("CPU:0")
    if CUDA_AVAILABLE:
        log.info("GPU-Modus aktiv (Open3D CUDA)")
    else:
        log.info("CPU-Modus (kein CUDA verfügbar)")
except ImportError:
    O3D_AVAILABLE = False
    CUDA_AVAILABLE = False
    DEVICE = None
    log.warning("open3d nicht installiert")


def to_gpu(pcd_legacy: object) -> object:
    """Konvertiert Legacy PointCloud → Tensor PointCloud auf GPU."""
    if not CUDA_AVAILABLE:
        return pcd_legacy
    try:
        t_pcd = o3d.t.geometry.PointCloud.from_legacy(pcd_legacy, device=DEVICE)
        return t_pcd
    except Exception as e:
        log.warning(f"GPU-Transfer fehlgeschlagen, nutze CPU: {e}")
        return pcd_legacy


def to_cpu(pcd) -> object:
    """Konvertiert Tensor PointCloud → Legacy PointCloud (für Algorithmen ohne GPU-Support)."""
    if hasattr(pcd, 'to_legacy'):
        return pcd.to_legacy()
    return pcd


def is_tensor_pcd(pcd) -> bool:
    return hasattr(pcd, 'to_legacy')


class PointCloudProcessor:

    def __init__(self, config: ProcessingConfig):
        self.cfg = config
        self._log_device()

    def _log_device(self):
        if CUDA_AVAILABLE:
            try:
                mem = o3d.core.cuda.device_count()
                log.info(f"CUDA-Geräte verfügbar: {mem}")
            except Exception:
                pass

    # ──────────────────────────────────────────
    # Vorverarbeitung (GPU-beschleunigt)
    # ──────────────────────────────────────────

    def preprocess(self, pcd_in: object, label: str = "") -> object:
        """
        Vollständige Vorverarbeitung mit GPU wo möglich:
        1. Voxel-Downsampling (GPU)
        2. Outlier-Entfernung (GPU: Radius, CPU-Fallback: SOR)
        3. Normalen-Schätzung (GPU)
        """
        if not O3D_AVAILABLE:
            return pcd_in

        n_start = len(pcd_in.points) if not is_tensor_pcd(pcd_in) else len(pcd_in.point.positions)
        log.info(f"Vorverarbeitung '{label}': {n_start:,} Punkte | GPU={CUDA_AVAILABLE}")
        t0 = time.time()

        if CUDA_AVAILABLE:
            result = self._preprocess_gpu(pcd_in)
        else:
            result = self._preprocess_cpu(pcd_in)

        n_end = len(result.points) if not is_tensor_pcd(result) else len(result.point.positions)
        log.info(f"  Vorverarbeitung fertig: {n_end:,} Punkte in {time.time()-t0:.1f}s")
        return result

    def _preprocess_gpu(self, pcd_legacy: object) -> object:
        """GPU-Pfad: Tensor API."""
        try:
            # Legacy → Tensor GPU
            t_pcd = o3d.t.geometry.PointCloud.from_legacy(pcd_legacy, device=DEVICE)

            # 1. Voxel Downsampling auf GPU
            t_pcd = t_pcd.voxel_down_sample(voxel_size=self.cfg.voxel_size_m)
            log.debug(f"  GPU Downsampling: {len(t_pcd.point.positions):,} Punkte")

            # 2. Normalen auf GPU
            t_pcd.estimate_normals(
                max_nn=30,
                radius=self.cfg.voxel_size_m * 2
            )
            log.debug("  GPU Normalen geschätzt")

            # 3. Zurück zu Legacy für SOR (kein CUDA-SOR in Open3D)
            pcd_legacy_clean = t_pcd.to_legacy()

            # SOR auf CPU (schnell nach Downsampling)
            pcd_clean, _ = pcd_legacy_clean.remove_statistical_outlier(
                nb_neighbors=self.cfg.sor_nb_neighbors,
                std_ratio=self.cfg.sor_std_ratio
            )
            removed = len(pcd_legacy_clean.points) - len(pcd_clean.points)
            log.info(f"  SOR (CPU): {removed:,} Ausreißer entfernt → {len(pcd_clean.points):,} Punkte")

            # Normalen orientieren
            pcd_clean.orient_normals_consistent_tangent_plane(100)
            return pcd_clean

        except Exception as e:
            log.warning(f"GPU-Vorverarbeitung fehlgeschlagen, Fallback CPU: {e}")
            return self._preprocess_cpu(pcd_legacy)

    def _preprocess_cpu(self, pcd: object) -> object:
        """CPU-Pfad: Legacy API."""
        pcd_down = pcd.voxel_down_sample(voxel_size=self.cfg.voxel_size_m)

        pcd_down.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=self.cfg.voxel_size_m * 2, max_nn=30
            )
        )
        pcd_down.orient_normals_consistent_tangent_plane(100)

        pcd_clean, _ = pcd_down.remove_statistical_outlier(
            nb_neighbors=self.cfg.sor_nb_neighbors,
            std_ratio=self.cfg.sor_std_ratio
        )
        removed = len(pcd_down.points) - len(pcd_clean.points)
        log.info(f"  SOR: {removed:,} Ausreißer entfernt → {len(pcd_clean.points):,} Punkte")
        return pcd_clean

    # ──────────────────────────────────────────
    # Registrierung (GPU-beschleunigt)
    # ──────────────────────────────────────────

    def register_multiple(self, point_clouds: List[object]) -> Optional[object]:
        if not O3D_AVAILABLE:
            return None
        if len(point_clouds) == 0:
            return None
        if len(point_clouds) == 1:
            return point_clouds[0]

        log.info(f"Registriere {len(point_clouds)} Scans | GPU={CUDA_AVAILABLE}")

        combined = point_clouds[0]
        total_registered = 1

        for i, pcd_target in enumerate(point_clouds[1:], start=2):
            log.info(f"Registriere Scan {i}/{len(point_clouds)}...")
            t0 = time.time()

            if CUDA_AVAILABLE:
                result_icp, transform = self._icp_gpu(pcd_target, combined)
            else:
                result_icp = self._icp_cpu_global(pcd_target, combined)
                transform  = result_icp.transformation

            fitness = result_icp.fitness if hasattr(result_icp, 'fitness') else 0
            rmse    = result_icp.inlier_rmse if hasattr(result_icp, 'inlier_rmse') else 0

            if fitness < 0.3:
                log.warning(f"  Scan {i}: schlechte Konvergenz (fitness={fitness:.3f}) – übersprungen")
                continue

            log.info(
                f"  ICP: fitness={fitness:.4f}, "
                f"RMSE={rmse*1000:.3f} mm in {time.time()-t0:.1f}s"
            )

            pcd_aligned = pcd_target.transform(transform)
            combined    = self._merge_clouds([combined, pcd_aligned])
            total_registered += 1

        log.info(f"Registrierung: {total_registered}/{len(point_clouds)} Scans → {len(combined.points):,} Punkte")

        # Abschluss-Downsampling
        if CUDA_AVAILABLE:
            try:
                t_pcd = o3d.t.geometry.PointCloud.from_legacy(combined, device=DEVICE)
                t_pcd = t_pcd.voxel_down_sample(voxel_size=self.cfg.voxel_size_m)
                combined = t_pcd.to_legacy()
            except Exception:
                combined = combined.voxel_down_sample(voxel_size=self.cfg.voxel_size_m)
        else:
            combined = combined.voxel_down_sample(voxel_size=self.cfg.voxel_size_m)

        combined.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=self.cfg.voxel_size_m * 2, max_nn=30
            )
        )
        return combined

    def _icp_gpu(self, source_legacy, target_legacy):
        """ICP auf GPU via Open3D Tensor API."""
        try:
            src = o3d.t.geometry.PointCloud.from_legacy(source_legacy, device=DEVICE)
            tgt = o3d.t.geometry.PointCloud.from_legacy(target_legacy, device=DEVICE)

            # Normalen sicherstellen
            if not src.point.normals.shape[0] if hasattr(src.point, 'normals') else True:
                src.estimate_normals(max_nn=30, radius=self.cfg.voxel_size_m * 2)
            if not tgt.point.normals.shape[0] if hasattr(tgt.point, 'normals') else True:
                tgt.estimate_normals(max_nn=30, radius=self.cfg.voxel_size_m * 2)

            # Grobe Ausrichtung (FPFH auf CPU, dann ICP auf GPU)
            initial_transform = self._global_registration_cpu(source_legacy, target_legacy)

            # Feines ICP auf GPU
            result = o3d.t.pipelines.registration.icp(
                src, tgt,
                max_correspondence_distance=self.cfg.icp_max_correspondence_m,
                init_source_to_target=o3c.Tensor(initial_transform, dtype=o3c.Dtype.Float64, device=DEVICE),
                estimation_method=o3d.t.pipelines.registration.TransformationEstimationPointToPlane(),
                criteria=o3d.t.pipelines.registration.ICPConvergenceCriteria(
                    relative_fitness=self.cfg.icp_relative_fitness,
                    relative_rmse=self.cfg.icp_relative_rmse,
                    max_iteration=self.cfg.icp_max_iteration
                )
            )

            transform = result.transformation.numpy()
            log.debug(f"  GPU ICP fitness={result.fitness:.4f}")
            return result, transform

        except Exception as e:
            log.warning(f"GPU ICP fehlgeschlagen, Fallback CPU: {e}")
            result = self._icp_cpu_global(source_legacy, target_legacy)
            return result, result.transformation

    def _global_registration_cpu(self, source, target) -> np.ndarray:
        """FPFH + RANSAC auf CPU für initiale Ausrichtung."""
        try:
            src_fpfh = self._compute_fpfh(source)
            tgt_fpfh = self._compute_fpfh(target)
            distance_threshold = self.cfg.voxel_size_m * 1.5

            result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
                source, target,
                src_fpfh, tgt_fpfh,
                mutual_filter=True,
                max_correspondence_distance=distance_threshold,
                estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
                ransac_n=3,
                checkers=[
                    o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                    o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
                ],
                criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(
                    self.cfg.ransac_max_iteration, self.cfg.ransac_confidence
                )
            )
            return result.transformation
        except Exception:
            return np.eye(4)

    def _icp_cpu_global(self, source, target):
        """Vollständiges ICP auf CPU (Fallback)."""
        initial = self._global_registration_cpu(source, target)
        return o3d.pipelines.registration.registration_icp(
            source, target,
            max_correspondence_distance=self.cfg.icp_max_correspondence_m,
            init=initial,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
                relative_fitness=self.cfg.icp_relative_fitness,
                relative_rmse=self.cfg.icp_relative_rmse,
                max_iteration=self.cfg.icp_max_iteration
            )
        )

    def _compute_fpfh(self, pcd):
        radius = self.cfg.fpfh_radius_m
        if not pcd.has_normals():
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30)
            )
        return o3d.pipelines.registration.compute_fpfh_feature(
            pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 5, max_nn=100)
        )

    def _merge_clouds(self, clouds):
        merged = o3d.geometry.PointCloud()
        for pcd in clouds:
            merged += pcd
        return merged

    # ──────────────────────────────────────────
    # Statistiken
    # ──────────────────────────────────────────

    def compute_stats(self, pcd: object) -> dict:
        if not O3D_AVAILABLE or pcd is None:
            return {}
        points = np.asarray(pcd.points)
        if len(points) == 0:
            return {}
        bbox   = pcd.get_axis_aligned_bounding_box()
        extent = bbox.get_extent()
        stats  = {
            "n_points":    len(points),
            "bbox_x_mm":   float(extent[0]) * 1000,
            "bbox_y_mm":   float(extent[1]) * 1000,
            "bbox_z_mm":   float(extent[2]) * 1000,
            "has_colors":  pcd.has_colors(),
            "has_normals": pcd.has_normals(),
            "gpu_used":    CUDA_AVAILABLE,
        }
        log.info(
            f"Punktwolken-Stats: {stats['n_points']:,} Punkte | "
            f"BBox: {stats['bbox_x_mm']:.1f} x {stats['bbox_y_mm']:.1f} x {stats['bbox_z_mm']:.1f} mm | "
            f"GPU={CUDA_AVAILABLE}"
        )
        return stats
