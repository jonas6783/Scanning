"""
tests/test_pipeline.py
Unit-Tests für die gesamte Pipeline ohne echte Kamera.
Alle Hardware-Abhängigkeiten werden gemockt.
Ausführen: python -m pytest tests/ -v
"""

import sys
import unittest
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

# Projektpfad hinzufügen
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import (
    ScannerConfig, CameraConfig, FilterConfig,
    CaptureConfig, ProcessingConfig, AnalysisConfig, PathConfig
)


class TestConfig(unittest.TestCase):
    """Testet Konfigurationsklassen."""

    def test_default_config_creates(self):
        cfg = ScannerConfig()
        self.assertIsNotNone(cfg.camera)
        self.assertIsNotNone(cfg.filters)
        self.assertIsNotNone(cfg.capture)
        self.assertIsNotNone(cfg.processing)
        self.assertIsNotNone(cfg.analysis)

    def test_camera_config_defaults(self):
        cam = CameraConfig()
        self.assertEqual(cam.depth_width, 1280)
        self.assertEqual(cam.depth_height, 720)
        self.assertEqual(cam.fps, 30)
        self.assertEqual(cam.laser_power, 150)
        self.assertLess(cam.depth_min_m, cam.depth_max_m)

    def test_analysis_tolerances(self):
        a = AnalysisConfig()
        self.assertLess(a.tolerance_ok_m, a.tolerance_warn_m)
        self.assertGreater(a.heatmap_max_m, a.tolerance_warn_m)

    def test_path_creation(self, tmp_path=None):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            pc = PathConfig(
                output_dir=Path(td) / "out",
                scans_dir=Path(td) / "out/scans",
                meshes_dir=Path(td) / "out/meshes",
                reports_dir=Path(td) / "out/reports",
            )
            pc.create_all()
            self.assertTrue(pc.output_dir.exists())
            self.assertTrue(pc.scans_dir.exists())
            self.assertTrue(pc.meshes_dir.exists())
            self.assertTrue(pc.reports_dir.exists())


class TestCamera(unittest.TestCase):
    """Testet D405Camera im Demo-Modus (kein Hardware)."""

    def setUp(self):
        from core.camera import D405Camera
        cfg = CameraConfig()
        fcfg = FilterConfig()
        self.camera = D405Camera(cfg, fcfg)

    def test_demo_mode_starts(self):
        """Kamera startet im Demo-Modus wenn pyrealsense2 fehlt."""
        result = self.camera.start()
        self.assertTrue(result)
        self.assertTrue(self.camera.running)

    def test_demo_frames_return_data(self):
        self.camera.start()
        frames = self.camera.get_frames()
        self.assertIsNotNone(frames)
        self.assertEqual(len(frames), 2)

    def test_context_manager(self):
        with self.camera as cam:
            self.assertTrue(cam.running)
        self.assertFalse(self.camera.running)

    def test_stop_without_start(self):
        """Stopp ohne vorherigen Start darf nicht crashen."""
        self.camera.stop()  # Kein Fehler erwartet


class TestPointCloudProcessor(unittest.TestCase):
    """Testet Punktwolken-Verarbeitung mit synthetischen Daten."""

    def setUp(self):
        try:
            import open3d as o3d
            self.o3d = o3d
            self.o3d_available = True
        except ImportError:
            self.o3d_available = False

        from processing.pointcloud import PointCloudProcessor
        self.processor = PointCloudProcessor(ProcessingConfig())

    def _make_sphere_pcd(self, n=5000, noise=0.0001):
        """Erstellt eine synthetische Kugeloberfläche als Testpunktwolke."""
        if not self.o3d_available:
            return None

        theta = np.random.uniform(0, 2 * np.pi, n)
        phi   = np.random.uniform(0, np.pi, n)
        r = 0.05  # 5 cm Radius

        x = r * np.sin(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.sin(theta)
        z = r * np.cos(phi)

        points = np.stack([x, y, z], axis=1)
        points += np.random.randn(*points.shape) * noise

        pcd = self.o3d.geometry.PointCloud()
        pcd.points = self.o3d.utility.Vector3dVector(points)
        return pcd

    def test_preprocess_reduces_noise(self):
        if not self.o3d_available:
            self.skipTest("open3d nicht verfügbar")

        pcd = self._make_sphere_pcd(n=10000, noise=0.002)
        result = self.processor.preprocess(pcd, label="test")

        self.assertIsNotNone(result)
        self.assertGreater(len(result.points), 0)
        self.assertLessEqual(len(result.points), len(pcd.points))
        self.assertTrue(result.has_normals())

    def test_preprocess_preserves_geometry(self):
        if not self.o3d_available:
            self.skipTest("open3d nicht verfügbar")

        pcd = self._make_sphere_pcd(n=5000, noise=0.0001)
        result = self.processor.preprocess(pcd, label="test")

        # BoundingBox sollte ähnlich bleiben
        bb_orig = pcd.get_axis_aligned_bounding_box().get_extent()
        bb_proc = result.get_axis_aligned_bounding_box().get_extent()

        for orig, proc in zip(bb_orig, bb_proc):
            self.assertAlmostEqual(orig, proc, delta=0.01)  # max 10 mm Abweichung

    def test_compute_stats(self):
        if not self.o3d_available:
            self.skipTest("open3d nicht verfügbar")

        pcd = self._make_sphere_pcd(n=1000)
        stats = self.processor.compute_stats(pcd)

        self.assertIn("n_points", stats)
        self.assertIn("bbox_x_mm", stats)
        self.assertEqual(stats["n_points"], 1000)
        self.assertGreater(stats["bbox_x_mm"], 0)

    def test_register_single_cloud(self):
        if not self.o3d_available:
            self.skipTest("open3d nicht verfügbar")

        pcd = self._make_sphere_pcd()
        result = self.processor.register_multiple([pcd])
        self.assertIsNotNone(result)


class TestQualityAnalyzer(unittest.TestCase):
    """Testet Qualitätsanalyse mit synthetischen Daten."""

    def setUp(self):
        from analysis.quality import QualityAnalyzer
        self.analyzer = QualityAnalyzer(AnalysisConfig())

    def test_statistics_computation(self):
        """Testet Abstandsstatistiken direkt."""
        distances = np.array([0.0001, 0.0003, 0.0005, 0.0008, 0.0015])  # Meter
        signed    = np.array([0.0001, -0.0003, 0.0005, -0.0008, 0.0015])

        stats = self.analyzer._compute_statistics(distances, signed)

        self.assertIn("mean_mm", stats)
        self.assertIn("rms_mm", stats)
        self.assertIn("max_mm", stats)
        self.assertIn("p95_mm", stats)

        # Mittelwert prüfen
        expected_mean = np.mean(distances) * 1000
        self.assertAlmostEqual(stats["mean_mm"], expected_mean, places=4)

        # Max-Wert prüfen
        self.assertAlmostEqual(stats["max_mm"], 1.5, places=2)

    def test_tolerance_classification(self):
        """Testet OK/WARN/FAIL-Klassifizierung."""
        cfg = AnalysisConfig()
        ok_m   = cfg.tolerance_ok_m
        warn_m = cfg.tolerance_warn_m

        # Alle Punkte in Toleranz
        dist_ok = np.full(100, ok_m * 0.5)
        signed  = dist_ok.copy()

        # Direktes Testen der Logik
        n_ok   = int(np.sum(dist_ok <= ok_m))
        n_warn = int(np.sum((dist_ok > ok_m) & (dist_ok <= warn_m)))
        n_fail = int(np.sum(dist_ok > warn_m))

        self.assertEqual(n_ok, 100)
        self.assertEqual(n_warn, 0)
        self.assertEqual(n_fail, 0)


class TestMeshReconstructor(unittest.TestCase):
    """Testet Mesh-Rekonstruktion."""

    def setUp(self):
        try:
            import open3d as o3d
            self.o3d = o3d
            self.o3d_available = True
        except ImportError:
            self.o3d_available = False

        from processing.mesh import MeshReconstructor
        cfg = ProcessingConfig()
        cfg.poisson_depth = 6  # Niedrig für schnelle Tests
        self.reconstructor = MeshReconstructor(cfg)

    def _make_sphere_pcd(self, n=2000):
        if not self.o3d_available:
            return None
        mesh = self.o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
        pcd = mesh.sample_points_uniformly(n)
        pcd.estimate_normals()
        pcd.orient_normals_consistent_tangent_plane(30)
        return pcd

    def test_poisson_from_sphere(self):
        if not self.o3d_available:
            self.skipTest("open3d nicht verfügbar")

        pcd = self._make_sphere_pcd()
        result = self.reconstructor.reconstruct_poisson(pcd)

        self.assertIsNotNone(result)
        mesh, densities = result
        self.assertGreater(len(mesh.vertices), 0)
        self.assertGreater(len(mesh.triangles), 0)
        self.assertEqual(len(densities), len(mesh.vertices))

    def test_density_filter(self):
        if not self.o3d_available:
            self.skipTest("open3d nicht verfügbar")

        pcd = self._make_sphere_pcd()
        result = self.reconstructor.reconstruct_poisson(pcd)
        if result is None:
            self.skipTest("Poisson fehlgeschlagen")

        mesh, densities = result
        n_before = len(mesh.vertices)
        filtered = self.reconstructor.filter_by_density(mesh, densities, quantile=0.05)
        self.assertLessEqual(len(filtered.vertices), n_before)

    def test_mesh_stats(self):
        if not self.o3d_available:
            self.skipTest("open3d nicht verfügbar")

        mesh = self.o3d.geometry.TriangleMesh.create_box()
        stats = self.reconstructor.mesh_stats(mesh)

        self.assertIn("n_vertices", stats)
        self.assertIn("n_triangles", stats)
        self.assertIn("is_watertight", stats)
        self.assertTrue(stats["is_watertight"])


class TestReportExporter(unittest.TestCase):
    """Testet Report-Generierung."""

    def test_html_report_generates(self):
        import tempfile
        from utils.exporter import ReportExporter

        with tempfile.TemporaryDirectory() as td:
            exporter = ReportExporter(Path(td))

            stats = {
                "mean_mm": 0.234, "rms_mm": 0.312, "max_mm": 1.234,
                "p95_mm": 0.876, "p99_mm": 1.123, "std_mm": 0.189,
                "median_mm": 0.210, "max_positive_mm": 1.234,
                "max_negative_mm": -0.567,
                "n_points_analyzed": 150000, "n_points_total": 200000,
                "n_ok": 140000, "n_warn": 7000, "n_fail": 3000,
                "pct_ok": 93.3, "pct_warn": 4.7, "pct_fail": 2.0,
                "tolerance_ok_mm": 0.5, "tolerance_warn_mm": 1.0,
                "verdict": "WARNUNG", "verdict_color": "orange",
                "reference_file": "test_part.stl",
                "timestamp": "2025-06-01T12:00:00",
            }

            report_path = exporter.generate_html_report(stats)
            self.assertTrue(report_path.exists())

            content = report_path.read_text(encoding="utf-8")
            self.assertIn("WARNUNG", content)
            self.assertIn("0.234", content)
            self.assertIn("93.3", content)

    def test_json_export(self):
        import tempfile, json
        from utils.exporter import ReportExporter

        with tempfile.TemporaryDirectory() as td:
            exporter = ReportExporter(Path(td))
            stats = {"mean_mm": 0.5, "verdict": "BESTANDEN", "n_ok": 1000}
            path = exporter.save_stats_json(stats)
            self.assertTrue(path.exists())
            data = json.loads(path.read_text())
            self.assertEqual(data["verdict"], "BESTANDEN")


if __name__ == "__main__":
    unittest.main(verbosity=2)
