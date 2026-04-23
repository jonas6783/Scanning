"""
processing/mesh.py
Mesh-Rekonstruktion aus Punktwolken via Poisson Surface Reconstruction.
Optimiert für Gussbauteile mit komplexer Geometrie.
"""

from pathlib import Path
from typing import Optional, Tuple
import numpy as np

from config.settings import ProcessingConfig
from utils.logger import setup_logger

log = setup_logger("mesh")

try:
    import open3d as o3d
    O3D_AVAILABLE = True
except ImportError:
    O3D_AVAILABLE = False


class MeshReconstructor:
    """
    Erstellt ein wasserdichtes 3D-Mesh aus einer Punktwolke.
    Verwendet Poisson Surface Reconstruction mit Qualitätsfilterung.
    """

    def __init__(self, config: ProcessingConfig):
        self.cfg = config

    # ──────────────────────────────────────────
    # Poisson-Rekonstruktion
    # ──────────────────────────────────────────

    def reconstruct_poisson(self, pcd: object) -> Optional[Tuple[object, np.ndarray]]:
        """
        Poisson Surface Reconstruction.
        Gibt (mesh, density_array) zurück.
        Dichte-Array wird für Qualitätsfilterung verwendet.

        Args:
            pcd: Open3D PointCloud (muss Normalen haben!)

        Returns:
            Tuple aus (TriangleMesh, density_array) oder None bei Fehler
        """
        if not O3D_AVAILABLE:
            log.warning("open3d nicht verfügbar")
            return None

        if pcd is None or len(pcd.points) == 0:
            log.error("Leere Punktwolke – Mesh-Rekonstruktion abgebrochen")
            return None

        if not pcd.has_normals():
            log.warning("Punktwolke hat keine Normalen – schätze nach...")
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(
                    radius=self.cfg.voxel_size_m * 2, max_nn=30
                )
            )
            pcd.orient_normals_consistent_tangent_plane(100)

        n_points = len(pcd.points)
        log.info(
            f"Starte Poisson-Rekonstruktion: {n_points:,} Punkte, "
            f"Depth={self.cfg.poisson_depth}"
        )

        try:
            mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                pcd,
                depth=self.cfg.poisson_depth,
                width=self.cfg.poisson_width,
                scale=self.cfg.poisson_scale,
                linear_fit=self.cfg.poisson_linear_fit
            )

            n_triangles = len(mesh.triangles)
            n_vertices = len(mesh.vertices)
            log.info(f"Rohes Mesh: {n_vertices:,} Vertices, {n_triangles:,} Dreiecke")

            return mesh, np.asarray(densities)

        except Exception as e:
            log.error(f"Poisson-Rekonstruktion fehlgeschlagen: {e}")
            return None

    def filter_by_density(
        self,
        mesh: object,
        densities: np.ndarray,
        quantile: Optional[float] = None
    ) -> object:
        """
        Entfernt dünn besetzte Mesh-Bereiche (typischerweise Artefakte
        an den Rändern der Punktwolke).

        Args:
            mesh:      Open3D TriangleMesh
            densities: Dichte-Array aus Poisson-Rekonstruktion
            quantile:  Unterste X% entfernen (None = Konfigurationswert)
        """
        if quantile is None:
            quantile = self.cfg.mesh_density_quantile

        threshold = np.quantile(densities, quantile)
        vertices_to_remove = densities < threshold

        mesh_clean = mesh
        mesh_clean.remove_vertices_by_mask(vertices_to_remove)

        removed = int(vertices_to_remove.sum())
        remaining = len(mesh_clean.vertices)

        log.info(
            f"Dichte-Filter (Q={quantile:.2f}): {removed:,} Vertices entfernt → "
            f"{remaining:,} verbleiben"
        )
        return mesh_clean

    def post_process(self, mesh: object) -> object:
        """
        Abschließende Mesh-Bereinigung:
        - Doppelte Vertices und Dreiecke entfernen
        - Nicht-manifold Kanten reparieren
        - Kleine isolierte Komponenten entfernen
        - Normalen neuberechnen
        """
        if not O3D_AVAILABLE or mesh is None:
            return mesh

        log.info("Mesh-Nachbearbeitung...")
        n_start = len(mesh.triangles)

        # Doppelte Geometrie entfernen
        mesh.remove_duplicated_vertices()
        mesh.remove_duplicated_triangles()
        mesh.remove_degenerate_triangles()
        mesh.remove_non_manifold_edges()

        # Kleine Komponenten entfernen (Artefakte)
        triangle_clusters, cluster_n_triangles, _ = mesh.cluster_connected_triangles()
        triangle_clusters = np.asarray(triangle_clusters)
        cluster_n_triangles = np.asarray(cluster_n_triangles)

        # Behalte nur die größte Komponente (das Bauteil selbst)
        largest_cluster = cluster_n_triangles.argmax()
        triangles_to_remove = triangle_clusters != largest_cluster
        mesh.remove_triangles_by_mask(triangles_to_remove)
        mesh.remove_unreferenced_vertices()

        # Normalen für korrekte Visualisierung
        mesh.compute_vertex_normals()
        mesh.compute_triangle_normals()

        n_end = len(mesh.triangles)
        log.info(
            f"Mesh bereinigt: {n_start:,} → {n_end:,} Dreiecke "
            f"({n_start - n_end:,} entfernt)"
        )

        return mesh

    # ──────────────────────────────────────────
    # Vollständiger Workflow
    # ──────────────────────────────────────────

    def build_mesh(self, pcd: object) -> Optional[object]:
        """
        Kompletter Mesh-Aufbau: Poisson → Dichte-Filter → Bereinigung.
        Gibt fertiges Mesh zurück oder None bei Fehler.
        """
        result = self.reconstruct_poisson(pcd)
        if result is None:
            return None

        mesh, densities = result
        mesh = self.filter_by_density(mesh, densities)
        mesh = self.post_process(mesh)

        # Qualitätsprüfung
        stats = self.mesh_stats(mesh)
        if stats["is_watertight"]:
            log.info("Mesh ist wasserdicht (watertight)")
        else:
            log.warning(
                "Mesh ist NICHT wasserdicht – Hinterschnitte oder Lücken in der Punktwolke. "
                "Mehr Scan-Positionen oder Scan-Spray aufbringen."
            )

        return mesh

    def mesh_stats(self, mesh: object) -> dict:
        """Berechnet Mesh-Qualitätsstatistiken."""
        if not O3D_AVAILABLE or mesh is None:
            return {}

        bbox = mesh.get_axis_aligned_bounding_box()
        extent = bbox.get_extent()

        stats = {
            "n_vertices":   len(mesh.vertices),
            "n_triangles":  len(mesh.triangles),
            "bbox_x_mm":    extent[0] * 1000,
            "bbox_y_mm":    extent[1] * 1000,
            "bbox_z_mm":    extent[2] * 1000,
            "is_watertight": mesh.is_watertight(),
            "is_orientable": mesh.is_orientable(),
        }

        log.info(
            f"Mesh-Stats: {stats['n_vertices']:,} Vertices, "
            f"{stats['n_triangles']:,} Dreiecke, "
            f"Watertight: {stats['is_watertight']}, "
            f"BBox: {stats['bbox_x_mm']:.1f}×{stats['bbox_y_mm']:.1f}×{stats['bbox_z_mm']:.1f} mm"
        )
        return stats

    # ──────────────────────────────────────────
    # Export
    # ──────────────────────────────────────────

    def save(self, mesh: object, path: Path, fmt: str = "stl") -> bool:
        """Speichert Mesh in gewähltem Format (stl/ply/obj)."""
        if not O3D_AVAILABLE or mesh is None:
            return False

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        suffix_map = {"stl": ".stl", "ply": ".ply", "obj": ".obj"}
        suffix = suffix_map.get(fmt.lower(), ".ply")
        save_path = path.with_suffix(suffix)

        try:
            success = o3d.io.write_triangle_mesh(str(save_path), mesh)
            if success:
                size_mb = save_path.stat().st_size / 1024 / 1024
                log.info(f"Mesh gespeichert: {save_path} ({size_mb:.1f} MB)")
            else:
                log.error(f"Mesh-Speichern fehlgeschlagen: {save_path}")
            return success
        except Exception as e:
            log.error(f"Mesh-Export Fehler: {e}")
            return False
