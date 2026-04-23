"""
processing/mesh.py
Mesh-Rekonstruktion mit GPU-Optimierungen wo möglich.

Poisson läuft auf CPU (kein CUDA-Support in Open3D).
Aber: Normalen-Schätzung und Vorbereitungsschritte laufen auf GPU.
"""

from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import time

from config.settings import ProcessingConfig
from utils.logger import setup_logger

log = setup_logger("mesh")

try:
    import open3d as o3d
    import open3d.core as o3c
    O3D_AVAILABLE = True
    CUDA_AVAILABLE = o3d.core.cuda.is_available()
    DEVICE = o3c.Device("CUDA:0") if CUDA_AVAILABLE else o3c.Device("CPU:0")
except ImportError:
    O3D_AVAILABLE = False
    CUDA_AVAILABLE = False
    DEVICE = None


class MeshReconstructor:

    def __init__(self, config: ProcessingConfig):
        self.cfg = config

    # ──────────────────────────────────────────
    # Normalen vorbereiten (GPU)
    # ──────────────────────────────────────────

    def _prepare_normals_gpu(self, pcd: object) -> object:
        """Normalen-Schätzung auf GPU, gibt Legacy PCD zurück."""
        try:
            t_pcd = o3d.t.geometry.PointCloud.from_legacy(pcd, device=DEVICE)
            t_pcd.estimate_normals(
                max_nn=30,
                radius=self.cfg.voxel_size_m * 2
            )
            pcd_out = t_pcd.to_legacy()
            pcd_out.orient_normals_consistent_tangent_plane(100)
            log.debug("Normalen auf GPU geschätzt")
            return pcd_out
        except Exception as e:
            log.warning(f"GPU Normalen fehlgeschlagen, CPU Fallback: {e}")
            return self._prepare_normals_cpu(pcd)

    def _prepare_normals_cpu(self, pcd: object) -> object:
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=self.cfg.voxel_size_m * 2, max_nn=30
            )
        )
        pcd.orient_normals_consistent_tangent_plane(100)
        return pcd

    # ──────────────────────────────────────────
    # Poisson Rekonstruktion (CPU – unvermeidbar)
    # ──────────────────────────────────────────

    def reconstruct_poisson(self, pcd: object) -> Optional[Tuple[object, np.ndarray]]:
        if not O3D_AVAILABLE or pcd is None or len(pcd.points) == 0:
            log.error("Leere Punktwolke oder open3d fehlt")
            return None

        # Normalen vorbereiten (GPU wenn möglich)
        if not pcd.has_normals():
            log.info("Schätze Normalen...")
            if CUDA_AVAILABLE:
                pcd = self._prepare_normals_gpu(pcd)
            else:
                pcd = self._prepare_normals_cpu(pcd)

        log.info(
            f"Poisson-Rekonstruktion (CPU): {len(pcd.points):,} Punkte | "
            f"Depth={self.cfg.poisson_depth}"
        )
        log.info("  Hinweis: Poisson läuft immer auf CPU – das ist normal")

        t0 = time.time()
        try:
            mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                pcd,
                depth=self.cfg.poisson_depth,
                width=self.cfg.poisson_width,
                scale=self.cfg.poisson_scale,
                linear_fit=self.cfg.poisson_linear_fit
            )
            elapsed = time.time() - t0
            log.info(
                f"Rohes Mesh: {len(mesh.vertices):,} Vertices, "
                f"{len(mesh.triangles):,} Dreiecke in {elapsed:.1f}s"
            )
            return mesh, np.asarray(densities)

        except Exception as e:
            log.error(f"Poisson fehlgeschlagen: {e}")
            return None

    # ──────────────────────────────────────────
    # Dichte-Filter + Bereinigung
    # ──────────────────────────────────────────

    def filter_by_density(self, mesh, densities, quantile=None) -> object:
        if quantile is None:
            quantile = self.cfg.mesh_density_quantile

        if len(densities) == 0 or len(mesh.vertices) == 0:
            log.error("Mesh leer – poisson_depth zu hoch oder zu wenige Punkte")
            return mesh

        threshold = np.quantile(densities, quantile)
        vertices_to_remove = densities < threshold
        mesh.remove_vertices_by_mask(vertices_to_remove)

        log.info(
            f"Dichte-Filter (Q={quantile:.2f}): {int(vertices_to_remove.sum()):,} Vertices entfernt → "
            f"{len(mesh.vertices):,} verbleiben"
        )
        return mesh

    def post_process(self, mesh) -> object:
        if not O3D_AVAILABLE or mesh is None:
            return mesh

        log.info("Mesh-Nachbearbeitung...")
        n_start = len(mesh.triangles)

        mesh.remove_duplicated_vertices()
        mesh.remove_duplicated_triangles()
        mesh.remove_degenerate_triangles()
        mesh.remove_non_manifold_edges()

        triangle_clusters, cluster_n, _ = mesh.cluster_connected_triangles()
        triangle_clusters = np.asarray(triangle_clusters)
        cluster_n         = np.asarray(cluster_n)

        if len(cluster_n) > 0:
            largest = cluster_n.argmax()
            mesh.remove_triangles_by_mask(triangle_clusters != largest)
            mesh.remove_unreferenced_vertices()

        mesh.compute_vertex_normals()
        mesh.compute_triangle_normals()

        log.info(
            f"Mesh bereinigt: {n_start:,} → {len(mesh.triangles):,} Dreiecke "
            f"({n_start - len(mesh.triangles):,} entfernt)"
        )
        return mesh

    # ──────────────────────────────────────────
    # Vollständiger Workflow
    # ──────────────────────────────────────────

    def build_mesh(self, pcd) -> Optional[object]:
        result = self.reconstruct_poisson(pcd)
        if result is None:
            return None

        mesh, densities = result
        mesh = self.filter_by_density(mesh, densities)
        mesh = self.post_process(mesh)

        stats = self.mesh_stats(mesh)
        if stats.get("is_watertight"):
            log.info("Mesh ist wasserdicht")
        else:
            log.warning("Mesh ist NICHT wasserdicht – mehr Scan-Positionen empfohlen")

        return mesh

    def mesh_stats(self, mesh) -> dict:
        if not O3D_AVAILABLE or mesh is None:
            return {}
        bbox   = mesh.get_axis_aligned_bounding_box()
        extent = bbox.get_extent()
        stats  = {
            "n_vertices":    len(mesh.vertices),
            "n_triangles":   len(mesh.triangles),
            "bbox_x_mm":     float(extent[0]) * 1000,
            "bbox_y_mm":     float(extent[1]) * 1000,
            "bbox_z_mm":     float(extent[2]) * 1000,
            "is_watertight": mesh.is_watertight(),
            "is_orientable": mesh.is_orientable(),
        }
        log.info(
            f"Mesh-Stats: {stats['n_vertices']:,} Vertices, "
            f"{stats['n_triangles']:,} Dreiecke, "
            f"Watertight: {stats['is_watertight']}, "
            f"BBox: {stats['bbox_x_mm']:.1f}x{stats['bbox_y_mm']:.1f}x{stats['bbox_z_mm']:.1f} mm"
        )
        return stats

    def save(self, mesh, path: Path, fmt: str = "stl") -> bool:
        if not O3D_AVAILABLE or mesh is None:
            return False
        path = Path(path).with_suffix({"stl": ".stl", "ply": ".ply", "obj": ".obj"}.get(fmt.lower(), ".ply"))
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            success = o3d.io.write_triangle_mesh(str(path), mesh)
            if success:
                size_mb = path.stat().st_size / 1024 / 1024
                log.info(f"Mesh gespeichert: {path} ({size_mb:.1f} MB)")
            return success
        except Exception as e:
            log.error(f"Mesh-Export Fehler: {e}")
            return False
