"""
analysis/quality.py
Qualitätskontrolle: CAD-Referenzvergleich, Cloud-to-Mesh-Abstandsanalyse,
Statistiken und Heatmap-Generierung.
"""

from pathlib import Path
from typing import Optional, Dict, Tuple
from datetime import datetime
import numpy as np

from config.settings import AnalysisConfig
from utils.logger import setup_logger

log = setup_logger("quality")

try:
    import open3d as o3d
    O3D_AVAILABLE = True
except ImportError:
    O3D_AVAILABLE = False

try:
    import matplotlib
    matplotlib.use("Agg")  # Kein Display nötig
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    MPL_AVAILABLE = True
except ImportError:
    MPL_AVAILABLE = False


class QualityAnalyzer:
    """
    Vergleicht Scan-Ergebnis mit CAD-Referenz.
    Berechnet Abstandsstatistiken und erstellt visuelle Reports.
    """

    def __init__(self, config: AnalysisConfig):
        self.cfg = config

    # ──────────────────────────────────────────
    # CAD-Referenzvergleich
    # ──────────────────────────────────────────

    def compare_to_reference(
        self,
        scan_pcd: object,
        reference_path: Path
    ) -> Optional[Dict]:
        """
        Vergleicht Scan-Punktwolke mit CAD-Referenz-Mesh.
        Berechnet Cloud-to-Mesh-Abstände für jeden Punkt.

        Args:
            scan_pcd:       Open3D PointCloud (gescanntes Bauteil)
            reference_path: Pfad zur CAD-Datei (.stl, .ply, .obj)

        Returns:
            Dict mit Abstandsstatistiken und Ergebnissen
        """
        if not O3D_AVAILABLE:
            log.warning("open3d nicht verfügbar")
            return None

        log.info(f"Lade CAD-Referenz: {reference_path}")
        reference_mesh = o3d.io.read_triangle_mesh(str(reference_path))

        if len(reference_mesh.triangles) == 0:
            log.error(f"CAD-Datei konnte nicht geladen werden: {reference_path}")
            return None

        log.info(
            f"CAD-Referenz geladen: "
            f"{len(reference_mesh.vertices):,} Vertices, "
            f"{len(reference_mesh.triangles):,} Dreiecke"
        )

        # Stichprobe für Performance (bei sehr großen Wolken)
        scan_points = scan_pcd
        n_scan = len(scan_pcd.points)
        if n_scan > self.cfg.analysis_sample_points:
            log.info(
                f"Stichprobe: {self.cfg.analysis_sample_points:,} von "
                f"{n_scan:,} Punkten für Analyse"
            )
            idx = np.random.choice(n_scan, self.cfg.analysis_sample_points, replace=False)
            scan_points = scan_pcd.select_by_index(idx.tolist())

        # CAD-Mesh als RaycastingScene für schnelle Abstands-Queries
        log.info("Berechne Cloud-to-Mesh Abstände...")
        reference_legacy = o3d.t.geometry.TriangleMesh.from_legacy(reference_mesh)
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(reference_legacy)

        # Abstände berechnen
        query_points = o3d.core.Tensor(
            np.asarray(scan_points.points), dtype=o3d.core.Dtype.Float32
        )
        distances = scene.compute_distance(query_points).numpy()

        # Vorzeichen (positiv = außerhalb CAD, negativ = innerhalb)
        # Verwende signed distance für CAD-Vergleich
        signed = scene.compute_signed_distance(query_points).numpy()

        log.info("Abstands-Berechnung abgeschlossen")

        # Statistiken
        stats = self._compute_statistics(distances, signed)
        stats["n_points_analyzed"] = len(distances)
        stats["n_points_total"]    = n_scan
        stats["reference_file"]    = str(reference_path)
        stats["timestamp"]         = datetime.now().isoformat()

        # Klassifizierung
        ok_m    = self.cfg.tolerance_ok_m
        warn_m  = self.cfg.tolerance_warn_m
        n_ok    = int(np.sum(distances <= ok_m))
        n_warn  = int(np.sum((distances > ok_m) & (distances <= warn_m)))
        n_fail  = int(np.sum(distances > warn_m))
        total   = len(distances)

        stats["pct_ok"]   = n_ok   / total * 100
        stats["pct_warn"] = n_warn / total * 100
        stats["pct_fail"] = n_fail / total * 100
        stats["n_ok"]     = n_ok
        stats["n_warn"]   = n_warn
        stats["n_fail"]   = n_fail

        # Gesamtbewertung
        if stats["pct_ok"] >= 95:
            stats["verdict"] = "BESTANDEN"
            stats["verdict_color"] = "green"
        elif stats["pct_fail"] <= 5:
            stats["verdict"] = "WARNUNG"
            stats["verdict_color"] = "orange"
        else:
            stats["verdict"] = "NICHT BESTANDEN"
            stats["verdict_color"] = "red"

        self._log_results(stats)

        # Punktwolke mit Abstandsfarben anreichern
        stats["colored_pcd"] = self._colorize_by_distance(scan_points, signed)
        stats["distances"]   = distances
        stats["signed_distances"] = signed

        return stats

    # ──────────────────────────────────────────
    # Statistiken
    # ──────────────────────────────────────────

    def _compute_statistics(
        self,
        distances: np.ndarray,
        signed: np.ndarray
    ) -> Dict:
        """Berechnet vollständige Abstandsstatistiken."""
        return {
            # Absolute Abstände (mm)
            "mean_mm":    float(np.mean(distances))   * 1000,
            "median_mm":  float(np.median(distances)) * 1000,
            "std_mm":     float(np.std(distances))    * 1000,
            "max_mm":     float(np.max(distances))    * 1000,
            "p95_mm":     float(np.percentile(distances, 95)) * 1000,
            "p99_mm":     float(np.percentile(distances, 99)) * 1000,
            "rms_mm":     float(np.sqrt(np.mean(distances**2))) * 1000,

            # Vorzeichenbehaftete Abstände (positiv = außerhalb, negativ = innen)
            "signed_mean_mm":   float(np.mean(signed))   * 1000,
            "signed_std_mm":    float(np.std(signed))    * 1000,
            "max_positive_mm":  float(np.max(signed))    * 1000,  # Material zu viel
            "max_negative_mm":  float(np.min(signed))    * 1000,  # Material fehlt

            # Toleranzen in mm (für Report)
            "tolerance_ok_mm":   self.cfg.tolerance_ok_m  * 1000,
            "tolerance_warn_mm": self.cfg.tolerance_warn_m * 1000,
        }

    def _log_results(self, stats: Dict):
        """Gibt Ergebniszusammenfassung im Log aus."""
        sep = "─" * 50
        log.info(sep)
        log.info(f"QUALITÄTSERGEBNIS: {stats['verdict']}")
        log.info(sep)
        log.info(f"  Punkte analysiert:  {stats['n_points_analyzed']:,}")
        log.info(f"  Mittlere Abweichung: {stats['mean_mm']:.3f} mm")
        log.info(f"  RMS-Abweichung:      {stats['rms_mm']:.3f} mm")
        log.info(f"  Max. Abweichung:     {stats['max_mm']:.3f} mm")
        log.info(f"  95. Perzentil:       {stats['p95_mm']:.3f} mm")
        log.info(sep)
        log.info(f"  Toleranz OK   (≤{stats['tolerance_ok_mm']:.1f} mm): {stats['pct_ok']:.1f}%")
        log.info(f"  Toleranz WARN (≤{stats['tolerance_warn_mm']:.1f} mm): {stats['pct_warn']:.1f}%")
        log.info(f"  AUSSERHALB:                     {stats['pct_fail']:.1f}%")
        log.info(sep)

    # ──────────────────────────────────────────
    # Visualisierung
    # ──────────────────────────────────────────

    def _colorize_by_distance(
        self,
        pcd: object,
        signed_distances: np.ndarray
    ) -> object:
        """
        Färbt Punktwolke nach Abweichung ein:
        Grün = OK, Gelb = Warnung, Rot = Fehler
        Blau = Material fehlt (negativ)
        """
        if not O3D_AVAILABLE:
            return pcd

        colored = o3d.geometry.PointCloud(pcd)
        colors = np.zeros((len(signed_distances), 3))

        ok_m   = self.cfg.tolerance_ok_m
        warn_m = self.cfg.tolerance_warn_m

        for i, d in enumerate(signed_distances):
            abs_d = abs(d)
            if abs_d <= ok_m:
                colors[i] = [0.0, 0.8, 0.2]   # Grün
            elif abs_d <= warn_m:
                # Interpolation: Grün → Gelb
                t = (abs_d - ok_m) / (warn_m - ok_m)
                colors[i] = [t, 0.8, 0.2 * (1-t)]
            else:
                if d > 0:
                    colors[i] = [0.9, 0.1, 0.1]   # Rot – Material zu viel
                else:
                    colors[i] = [0.1, 0.1, 0.9]   # Blau – Material fehlt

        colored.colors = o3d.utility.Vector3dVector(colors)
        return colored

    def save_heatmap(
        self,
        stats: Dict,
        output_path: Path
    ) -> Optional[Path]:
        """
        Erstellt und speichert eine Abstandsverteilungs-Heatmap als PNG.
        """
        if not MPL_AVAILABLE or "distances" not in stats:
            log.warning("matplotlib nicht verfügbar – Heatmap übersprungen")
            return None

        distances_mm = stats["distances"] * 1000
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.patch.set_facecolor("#1a1a2e")

        for ax in axes:
            ax.set_facecolor("#16213e")
            ax.tick_params(colors="white")
            ax.xaxis.label.set_color("white")
            ax.yaxis.label.set_color("white")
            ax.title.set_color("white")
            for spine in ax.spines.values():
                spine.set_edgecolor("#444")

        # Histogramm – Abstandsverteilung
        ax = axes[0]
        n, bins, patches = ax.hist(
            distances_mm, bins=100,
            color="#378ADD", edgecolor="none", alpha=0.85
        )
        # Farbe nach Toleranz
        for patch, left in zip(patches, bins[:-1]):
            if left <= stats["tolerance_ok_mm"]:
                patch.set_facecolor("#1D9E75")
            elif left <= stats["tolerance_warn_mm"]:
                patch.set_facecolor("#EF9F27")
            else:
                patch.set_facecolor("#E24B4A")

        ax.axvline(stats["tolerance_ok_mm"],  color="#1D9E75", linestyle="--", linewidth=1.5,
                   label=f"OK ≤{stats['tolerance_ok_mm']:.1f} mm")
        ax.axvline(stats["tolerance_warn_mm"], color="#EF9F27", linestyle="--", linewidth=1.5,
                   label=f"WARN ≤{stats['tolerance_warn_mm']:.1f} mm")
        ax.axvline(stats["mean_mm"], color="white", linestyle=":", linewidth=1,
                   label=f"Mittelwert {stats['mean_mm']:.3f} mm")

        ax.set_xlabel("Abweichung [mm]")
        ax.set_ylabel("Anzahl Punkte")
        ax.set_title("Abstandsverteilung")
        ax.legend(fontsize=8, labelcolor="white", facecolor="#1a1a2e", edgecolor="#444")

        # Kreisdiagramm – OK / WARN / FAIL
        ax2 = axes[1]
        sizes  = [stats["pct_ok"], stats["pct_warn"], stats["pct_fail"]]
        labels = [
            f"OK ({stats['pct_ok']:.1f}%)",
            f"Warnung ({stats['pct_warn']:.1f}%)",
            f"Fehler ({stats['pct_fail']:.1f}%)"
        ]
        colors_pie = ["#1D9E75", "#EF9F27", "#E24B4A"]
        explode    = (0.03, 0.03, 0.05)

        wedges, texts, autotexts = ax2.pie(
            sizes, labels=labels, colors=colors_pie,
            explode=explode, autopct="%1.1f%%",
            startangle=90, textprops={"color": "white", "fontsize": 9}
        )
        for at in autotexts:
            at.set_color("white")
            at.set_fontsize(8)

        ax2.set_title(f"Qualitätsergebnis: {stats['verdict']}")

        # Statistik-Text
        info = (
            f"Punkte: {stats['n_points_analyzed']:,}  │  "
            f"Mittelwert: {stats['mean_mm']:.3f} mm  │  "
            f"RMS: {stats['rms_mm']:.3f} mm  │  "
            f"Max: {stats['max_mm']:.3f} mm  │  "
            f"P95: {stats['p95_mm']:.3f} mm"
        )
        fig.text(0.5, 0.02, info, ha="center", fontsize=8, color="#aaa",
                 bbox=dict(boxstyle="round", facecolor="#0f3460", alpha=0.5))

        fig.suptitle(
            f"{self.cfg.report_title}  –  {stats['timestamp'][:10]}",
            fontsize=12, color="white", fontweight="bold"
        )

        plt.tight_layout(rect=[0, 0.06, 1, 0.95])

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(output_path), dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close()

        log.info(f"Heatmap gespeichert: {output_path}")
        return output_path
