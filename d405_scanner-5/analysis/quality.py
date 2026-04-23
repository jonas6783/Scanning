"""
analysis/quality.py
Qualitätskontrolle: CAD-Referenzvergleich, Zonen-Toleranzen,
Cloud-to-Mesh-Abstände, Heatmap und 3D-HTML-Visualisierung.
"""

from pathlib import Path
from typing import Optional, Dict, List
from datetime import datetime
import numpy as np

from config.settings import AnalysisConfig, ToleranceZone
from utils.logger import setup_logger

log = setup_logger("quality")

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

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    MPL_AVAILABLE = True
except ImportError:
    MPL_AVAILABLE = False


class QualityAnalyzer:

    def __init__(self, config: AnalysisConfig):
        self.cfg = config

    # ──────────────────────────────────────────
    # CAD-Vergleich Hauptmethode
    # ──────────────────────────────────────────

    def compare_to_reference(self, scan_pcd, reference_path: Path) -> Optional[Dict]:
        if not O3D_AVAILABLE:
            return None

        log.info(f"Lade CAD-Referenz: {reference_path}")
        reference_mesh = o3d.io.read_triangle_mesh(str(reference_path))
        if len(reference_mesh.triangles) == 0:
            log.error(f"CAD konnte nicht geladen werden: {reference_path}")
            return None

        log.info(f"CAD: {len(reference_mesh.vertices):,} Vertices, {len(reference_mesh.triangles):,} Dreiecke")

        # Einheiten automatisch erkennen
        scan_extent = scan_pcd.get_axis_aligned_bounding_box().get_extent()
        cad_verts   = np.asarray(reference_mesh.vertices)
        cad_extent  = cad_verts.max(axis=0) - cad_verts.min(axis=0)
        scan_max    = float(np.max(scan_extent))
        cad_max     = float(np.max(cad_extent))

        log.info(f"Scan BBox max: {scan_max*1000:.1f} mm | CAD BBox max: {cad_max:.3f}")
        if cad_max > scan_max * 100:
            log.info("CAD in mm erkannt -> skaliere auf Meter (*0.001)")
            reference_mesh.scale(0.001, center=reference_mesh.get_center())
        elif cad_max < scan_max * 0.01:
            log.info("CAD sehr klein -> skaliere hoch (*1000)")
            reference_mesh.scale(1000.0, center=reference_mesh.get_center())
        else:
            log.info("CAD-Einheit passt")

        # Zentren angleichen
        scan_center = scan_pcd.get_axis_aligned_bounding_box().get_center()
        cad_center  = reference_mesh.get_axis_aligned_bounding_box().get_center()
        reference_mesh.translate(scan_center - cad_center)
        log.info("CAD auf Scan-Zentrum ausgerichtet")

        # Stichprobe
        scan_points = scan_pcd
        n_scan = len(scan_pcd.points)
        if n_scan > self.cfg.analysis_sample_points:
            idx = np.random.choice(n_scan, self.cfg.analysis_sample_points, replace=False)
            scan_points = scan_pcd.select_by_index(idx.tolist())
            log.info(f"Stichprobe: {self.cfg.analysis_sample_points:,} von {n_scan:,} Punkten")

        # Abstände berechnen
        log.info("Berechne Cloud-to-Mesh Abstände...")
        reference_legacy = o3d.t.geometry.TriangleMesh.from_legacy(reference_mesh)
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(reference_legacy)

        query_points = o3c.Tensor(
            np.asarray(scan_points.points), dtype=o3c.Dtype.Float32
        )
        # RaycastingScene läuft auf CPU – query_points werden intern konvertiert
        distances = scene.compute_distance(query_points).numpy()
        signed    = scene.compute_signed_distance(query_points).numpy()
        log.info("Abstands-Berechnung abgeschlossen")

        # Zonen-Klassifizierung
        points_arr = np.asarray(scan_points.points)
        centroid   = points_arr.mean(axis=0)
        zone_ids   = self._assign_zones(points_arr, centroid)
        zone_stats = self._compute_zone_stats(distances, signed, zone_ids)

        # Gesamt-Statistiken
        stats = self._compute_statistics(distances, signed)
        stats["n_points_analyzed"] = int(len(distances))
        stats["n_points_total"]    = int(n_scan)
        stats["reference_file"]    = str(reference_path)
        stats["timestamp"]         = datetime.now().isoformat()
        stats["zone_stats"]        = zone_stats

        ok_m   = self.cfg.tolerance_ok_m
        warn_m = self.cfg.tolerance_warn_m
        n_ok   = int(np.sum(distances <= ok_m))
        n_warn = int(np.sum((distances > ok_m) & (distances <= warn_m)))
        n_fail = int(np.sum(distances > warn_m))
        total  = len(distances)

        stats["pct_ok"]   = float(n_ok   / total * 100)
        stats["pct_warn"] = float(n_warn / total * 100)
        stats["pct_fail"] = float(n_fail / total * 100)
        stats["n_ok"]     = n_ok
        stats["n_warn"]   = n_warn
        stats["n_fail"]   = n_fail

        # Gesamturteil – auch Zonen-Fehler einbeziehen
        all_passed = stats["pct_ok"] >= 95 and all(
            z.get("verdict") in ("BESTANDEN", "OK") for z in zone_stats.values()
        )
        any_zone_failed = any(
            z.get("pct_fail", 0) > 5 for z in zone_stats.values()
        )

        if all_passed:
            stats["verdict"] = "BESTANDEN"
        elif stats["pct_fail"] <= 5 and not any_zone_failed:
            stats["verdict"] = "WARNUNG"
        else:
            stats["verdict"] = "NICHT BESTANDEN"

        self._log_results(stats)

        # Farbige Punktwolke (nach Zonen + Abweichung)
        stats["colored_pcd"]      = self._colorize_by_zone(scan_points, signed, zone_ids)
        stats["distances"]        = distances
        stats["signed_distances"] = signed
        stats["points_array"]     = points_arr
        stats["zone_ids"]         = zone_ids

        return stats

    # ──────────────────────────────────────────
    # Zonen
    # ──────────────────────────────────────────

    def _assign_zones(self, points: np.ndarray, centroid: np.ndarray) -> np.ndarray:
        """
        Weist jedem Punkt eine Zonen-ID zu.
        -1 = keine Zone (Default-Toleranz)
        """
        zone_ids = np.full(len(points), -1, dtype=int)
        rel = points - centroid  # Koordinaten relativ zum Zentrum

        for zone_idx, zone in enumerate(self.cfg.zones):
            mask = np.ones(len(points), dtype=bool)
            if zone.x_min is not None: mask &= rel[:, 0] >= zone.x_min
            if zone.x_max is not None: mask &= rel[:, 0] <= zone.x_max
            if zone.y_min is not None: mask &= rel[:, 1] >= zone.y_min
            if zone.y_max is not None: mask &= rel[:, 1] <= zone.y_max
            if zone.z_min is not None: mask &= rel[:, 2] >= zone.z_min
            if zone.z_max is not None: mask &= rel[:, 2] <= zone.z_max
            # Zonen überschreiben sich (letzte Definition gewinnt)
            zone_ids[mask] = zone_idx

        assigned = int(np.sum(zone_ids >= 0))
        log.info(f"Zonen-Zuweisung: {assigned:,} Punkte in Zonen, "
                 f"{len(points)-assigned:,} mit Default-Toleranz")
        return zone_ids

    def _compute_zone_stats(self, distances, signed, zone_ids) -> Dict:
        """Berechnet Statistiken pro Zone + Default."""
        result = {}

        # Default-Zone (keine spezifische Zone)
        default_mask = zone_ids == -1
        if default_mask.sum() > 0:
            d = distances[default_mask]
            s = signed[default_mask]
            ok_m   = self.cfg.tolerance_ok_m
            warn_m = self.cfg.tolerance_warn_m
            n_ok   = int(np.sum(d <= ok_m))
            n_warn = int(np.sum((d > ok_m) & (d <= warn_m)))
            n_fail = int(np.sum(d > warn_m))
            total  = len(d)
            pct_ok = float(n_ok / total * 100) if total > 0 else 0
            result["Default"] = {
                "name":          "Default (keine Zone)",
                "n_points":      total,
                "mean_mm":       float(np.mean(d)) * 1000,
                "max_mm":        float(np.max(d))  * 1000,
                "p95_mm":        float(np.percentile(d, 95)) * 1000,
                "pct_ok":        pct_ok,
                "pct_warn":      float(n_warn / total * 100),
                "pct_fail":      float(n_fail / total * 100),
                "tolerance_ok_mm":   ok_m   * 1000,
                "tolerance_warn_mm": warn_m * 1000,
                "verdict":       "BESTANDEN" if pct_ok >= 95 else ("WARNUNG" if n_fail/total <= 0.05 else "NICHT BESTANDEN"),
            }

        for zone_idx, zone in enumerate(self.cfg.zones):
            mask = zone_ids == zone_idx
            if mask.sum() == 0:
                log.warning(f"Zone '{zone.name}': keine Punkte – Koordinaten prüfen!")
                continue
            d = distances[mask]
            s = signed[mask]
            ok_m   = zone.tolerance_ok_m
            warn_m = zone.tolerance_warn_m
            n_ok   = int(np.sum(d <= ok_m))
            n_warn = int(np.sum((d > ok_m) & (d <= warn_m)))
            n_fail = int(np.sum(d > warn_m))
            total  = len(d)
            pct_ok = float(n_ok / total * 100) if total > 0 else 0
            verdict = "BESTANDEN" if pct_ok >= 95 else ("WARNUNG" if n_fail/total <= 0.05 else "NICHT BESTANDEN")
            result[zone.name] = {
                "name":          zone.name,
                "n_points":      total,
                "mean_mm":       float(np.mean(d)) * 1000,
                "max_mm":        float(np.max(d))  * 1000,
                "p95_mm":        float(np.percentile(d, 95)) * 1000,
                "pct_ok":        pct_ok,
                "pct_warn":      float(n_warn / total * 100),
                "pct_fail":      float(n_fail / total * 100),
                "tolerance_ok_mm":   ok_m   * 1000,
                "tolerance_warn_mm": warn_m * 1000,
                "verdict":       verdict,
            }
            log.info(f"  Zone '{zone.name}': {total:,} Punkte | "
                     f"OK {pct_ok:.1f}% | Mittel {float(np.mean(d))*1000:.3f} mm | {verdict}")

        return result

    def _colorize_by_zone(self, pcd, signed_distances, zone_ids) -> object:
        """Färbt Punkte nach Zone und Abweichung ein."""
        if not O3D_AVAILABLE:
            return pcd

        colored = o3d.geometry.PointCloud(pcd)
        colors  = np.zeros((len(signed_distances), 3))

        for i, (d, zone_id) in enumerate(zip(signed_distances, zone_ids)):
            abs_d = abs(d)

            if zone_id >= 0:
                zone   = self.cfg.zones[zone_id]
                ok_m   = zone.tolerance_ok_m
                warn_m = zone.tolerance_warn_m
                c_ok   = zone.color_ok
                c_warn = zone.color_warn
                c_fail = zone.color_fail
            else:
                ok_m   = self.cfg.tolerance_ok_m
                warn_m = self.cfg.tolerance_warn_m
                c_ok   = [0.0, 0.8, 0.2]
                c_warn = [1.0, 0.6, 0.0]
                c_fail = [0.9, 0.1, 0.1] if d > 0 else [0.1, 0.1, 0.9]

            if abs_d <= ok_m:
                colors[i] = c_ok
            elif abs_d <= warn_m:
                t = (abs_d - ok_m) / (warn_m - ok_m)
                colors[i] = [
                    c_ok[0] * (1 - t) + c_warn[0] * t,
                    c_ok[1] * (1 - t) + c_warn[1] * t,
                    c_ok[2] * (1 - t) + c_warn[2] * t,
                ]
            else:
                colors[i] = c_fail

        colored.colors = o3d.utility.Vector3dVector(colors)
        return colored

    # ──────────────────────────────────────────
    # Statistiken
    # ──────────────────────────────────────────

    def _compute_statistics(self, distances, signed) -> Dict:
        return {
            "mean_mm":           float(np.mean(distances))             * 1000,
            "median_mm":         float(np.median(distances))           * 1000,
            "std_mm":            float(np.std(distances))              * 1000,
            "max_mm":            float(np.max(distances))              * 1000,
            "p95_mm":            float(np.percentile(distances, 95))   * 1000,
            "p99_mm":            float(np.percentile(distances, 99))   * 1000,
            "rms_mm":            float(np.sqrt(np.mean(distances**2))) * 1000,
            "signed_mean_mm":    float(np.mean(signed))                * 1000,
            "signed_std_mm":     float(np.std(signed))                 * 1000,
            "max_positive_mm":   float(np.max(signed))                 * 1000,
            "max_negative_mm":   float(np.min(signed))                 * 1000,
            "tolerance_ok_mm":   float(self.cfg.tolerance_ok_m)        * 1000,
            "tolerance_warn_mm": float(self.cfg.tolerance_warn_m)      * 1000,
        }

    def _log_results(self, stats):
        sep = "─" * 50
        log.info(sep)
        log.info(f"QUALITÄTSERGEBNIS: {stats['verdict']}")
        log.info(sep)
        log.info(f"  Punkte analysiert:   {stats['n_points_analyzed']:,}")
        log.info(f"  Mittlere Abweichung: {stats['mean_mm']:.3f} mm")
        log.info(f"  RMS-Abweichung:      {stats['rms_mm']:.3f} mm")
        log.info(f"  Max. Abweichung:     {stats['max_mm']:.3f} mm")
        log.info(f"  95. Perzentil:       {stats['p95_mm']:.3f} mm")
        log.info(sep)
        log.info(f"  Toleranz OK   (<={stats['tolerance_ok_mm']:.1f} mm): {stats['pct_ok']:.1f}%")
        log.info(f"  Toleranz WARN (<={stats['tolerance_warn_mm']:.1f} mm): {stats['pct_warn']:.1f}%")
        log.info(f"  AUSSERHALB:                      {stats['pct_fail']:.1f}%")
        log.info(sep)

    # ──────────────────────────────────────────
    # Heatmap (PNG)
    # ──────────────────────────────────────────

    def save_heatmap(self, stats: Dict, output_path: Path) -> Optional[Path]:
        if not MPL_AVAILABLE or "distances" not in stats:
            return None

        distances_mm = stats["distances"] * 1000
        zone_stats   = stats.get("zone_stats", {})
        n_zones      = len(zone_stats)

        fig = plt.figure(figsize=(16, 5 + n_zones * 1.2))
        fig.patch.set_facecolor("#0f1117")

        # Layout: oben Histogramm + Pie, unten Zonen-Tabelle
        gs = fig.add_gridspec(2, 2, height_ratios=[3, max(1, n_zones * 0.8)], hspace=0.4)

        # ── Histogramm ──
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.set_facecolor("#16213e")
        for spine in ax1.spines.values(): spine.set_edgecolor("#333")
        ax1.tick_params(colors="white")
        ax1.xaxis.label.set_color("white")
        ax1.yaxis.label.set_color("white")
        ax1.title.set_color("white")

        n, bins, patches = ax1.hist(distances_mm, bins=120, edgecolor="none", alpha=0.9)
        for patch, left in zip(patches, bins[:-1]):
            if left <= stats["tolerance_ok_mm"]:
                patch.set_facecolor("#1D9E75")
            elif left <= stats["tolerance_warn_mm"]:
                patch.set_facecolor("#EF9F27")
            else:
                patch.set_facecolor("#E24B4A")

        ax1.axvline(stats["tolerance_ok_mm"],  color="#1D9E75", ls="--", lw=1.5, label=f"OK <={stats['tolerance_ok_mm']:.1f}mm")
        ax1.axvline(stats["tolerance_warn_mm"], color="#EF9F27", ls="--", lw=1.5, label=f"WARN <={stats['tolerance_warn_mm']:.1f}mm")
        ax1.axvline(stats["mean_mm"], color="white", ls=":", lw=1, label=f"Mittel {stats['mean_mm']:.3f}mm")
        ax1.set_xlabel("Abweichung [mm]")
        ax1.set_ylabel("Punkte")
        ax1.set_title("Abstandsverteilung (gesamt)")
        ax1.legend(fontsize=8, labelcolor="white", facecolor="#0f1117", edgecolor="#333")

        # ── Kreisdiagramm ──
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.set_facecolor("#16213e")
        ax2.title.set_color("white")
        sizes  = [stats["pct_ok"], stats["pct_warn"], stats["pct_fail"]]
        labels = [f"OK ({stats['pct_ok']:.1f}%)", f"Warn ({stats['pct_warn']:.1f}%)", f"Fehler ({stats['pct_fail']:.1f}%)"]
        wedges, texts, autotexts = ax2.pie(
            sizes, labels=labels,
            colors=["#1D9E75", "#EF9F27", "#E24B4A"],
            explode=(0.03, 0.03, 0.05), autopct="%1.1f%%",
            startangle=90, textprops={"color": "white", "fontsize": 9}
        )
        for at in autotexts: at.set_color("white"); at.set_fontsize(8)
        ax2.set_title(f"Ergebnis: {stats['verdict']}")

        # ── Zonen-Tabelle ──
        if zone_stats:
            ax3 = fig.add_subplot(gs[1, :])
            ax3.set_facecolor("#0f1117")
            ax3.axis("off")
            ax3.set_title("Zonen-Auswertung", color="white", fontsize=10, pad=6)

            col_labels = ["Zone", "Punkte", "Toleranz OK", "Toleranz WARN", "Mittel", "Max", "P95", "OK%", "FAIL%", "Ergebnis"]
            rows = []
            row_colors = []
            for zname, zs in zone_stats.items():
                v = zs["verdict"]
                vc = "#1D9E75" if "BESTANDEN" in v else ("#EF9F27" if "WARNUNG" in v else "#E24B4A")
                rows.append([
                    zname,
                    f"{zs['n_points']:,}",
                    f"{zs['tolerance_ok_mm']:.2f} mm",
                    f"{zs['tolerance_warn_mm']:.2f} mm",
                    f"{zs['mean_mm']:.3f} mm",
                    f"{zs['max_mm']:.3f} mm",
                    f"{zs['p95_mm']:.3f} mm",
                    f"{zs['pct_ok']:.1f}%",
                    f"{zs['pct_fail']:.1f}%",
                    v,
                ])
                row_colors.append(["#1a1f2e"] * 9 + [vc + "44"])

            tbl = ax3.table(
                cellText=rows, colLabels=col_labels,
                loc="center", cellLoc="center"
            )
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(8)
            tbl.scale(1, 1.4)
            for (r, c), cell in tbl.get_celld().items():
                cell.set_facecolor("#1a1f2e" if r > 0 else "#0d1b2a")
                cell.set_text_props(color="white")
                cell.set_edgecolor("#333")
                if r > 0 and c == len(col_labels) - 1:
                    v = rows[r - 1][-1]
                    cell.set_facecolor("#1D9E7544" if "BESTANDEN" in v else ("#EF9F2744" if "WARNUNG" in v else "#E24B4A44"))

        fig.suptitle(
            f"{self.cfg.report_title}  -  {stats['timestamp'][:10]}  |  Ergebnis: {stats['verdict']}",
            fontsize=12, color="white", fontweight="bold"
        )

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(output_path), dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close()
        log.info(f"Heatmap gespeichert: {output_path}")
        return output_path

    # ──────────────────────────────────────────
    # 3D HTML-Visualisierung
    # ──────────────────────────────────────────

    def save_3d_visualization(self, stats: Dict, output_path: Path) -> Optional[Path]:
        """
        Erstellt eine interaktive 3D-HTML-Ansicht der Abweichungen.
        Punkte sind nach Toleranz eingefärbt, Zonen sichtbar.
        Öffnet direkt im Browser – kein CloudCompare nötig.
        """
        try:
            import json as _json
        except ImportError:
            return None

        if "points_array" not in stats or "distances" not in stats:
            log.warning("Keine Punktdaten für 3D-Visualisierung")
            return None

        pts      = stats["points_array"]
        dists    = stats["distances"]
        signed   = stats["signed_distances"]
        zone_ids = stats.get("zone_ids", np.full(len(pts), -1))

        # Punkte auf max 50k reduzieren für Browser-Performance
        max_pts = 50_000
        if len(pts) > max_pts:
            idx = np.random.choice(len(pts), max_pts, replace=False)
            pts      = pts[idx]
            dists    = dists[idx]
            signed   = signed[idx]
            zone_ids = zone_ids[idx]

        # Farben berechnen (RGB 0-255)
        colors_rgb = []
        for d, s, zid in zip(dists, signed, zone_ids):
            abs_d = abs(d)
            if zid >= 0 and zid < len(self.cfg.zones):
                zone   = self.cfg.zones[zid]
                ok_m   = zone.tolerance_ok_m
                warn_m = zone.tolerance_warn_m
            else:
                ok_m   = self.cfg.tolerance_ok_m
                warn_m = self.cfg.tolerance_warn_m

            if abs_d <= ok_m:
                colors_rgb.append([0, 200, 60])
            elif abs_d <= warn_m:
                t = (abs_d - ok_m) / (warn_m - ok_m)
                colors_rgb.append([int(255 * t), int(200 * (1 - t) + 150 * t), 0])
            else:
                colors_rgb.append([230, 30, 30] if s > 0 else [30, 30, 220])

        colors_rgb = np.array(colors_rgb)

        # Daten als JSON
        data = {
            "x": pts[:, 0].tolist(),
            "y": pts[:, 1].tolist(),
            "z": pts[:, 2].tolist(),
            "r": colors_rgb[:, 0].tolist(),
            "g": colors_rgb[:, 1].tolist(),
            "b": colors_rgb[:, 2].tolist(),
            "d": (dists * 1000).round(3).tolist(),
            "s": (signed * 1000).round(3).tolist(),
        }

        # Zonen-Boxen für Visualisierung
        centroid = np.asarray(stats["points_array"]).mean(axis=0)
        zone_boxes = []
        for zone in self.cfg.zones:
            def r(v, fallback): return float(v) + float(centroid[{"x": 0, "y": 1, "z": 2}[fallback[0]]]) if v is not None else None
            zone_boxes.append({
                "name": zone.name,
                "ok_mm": zone.tolerance_ok_m * 1000,
                "warn_mm": zone.tolerance_warn_m * 1000,
            })

        # Statistiken für Overlay
        overview = {
            "verdict":   stats["verdict"],
            "mean_mm":   round(stats["mean_mm"], 3),
            "rms_mm":    round(stats["rms_mm"], 3),
            "max_mm":    round(stats["max_mm"], 3),
            "p95_mm":    round(stats["p95_mm"], 3),
            "pct_ok":    round(stats["pct_ok"], 1),
            "pct_warn":  round(stats["pct_warn"], 1),
            "pct_fail":  round(stats["pct_fail"], 1),
            "n_points":  stats["n_points_analyzed"],
            "timestamp": stats["timestamp"][:19],
            "reference": Path(stats["reference_file"]).name,
        }

        zone_stats_clean = {}
        for k, v in stats.get("zone_stats", {}).items():
            zone_stats_clean[k] = {kk: vv for kk, vv in v.items() if isinstance(vv, (str, int, float))}

        html = self._build_3d_html(data, overview, zone_stats_clean, zone_boxes)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(html, encoding="utf-8")
        log.info(f"3D-Visualisierung gespeichert: {output_path}")
        return output_path

    def _build_3d_html(self, data, overview, zone_stats, zone_boxes) -> str:
        import json as _json

        verdict_color = {"BESTANDEN": "#1D9E75", "WARNUNG": "#EF9F27", "NICHT BESTANDEN": "#E24B4A"}.get(
            overview["verdict"], "#888"
        )

        zone_rows = ""
        for zname, zs in zone_stats.items():
            v = zs.get("verdict", "–")
            vc = "#1D9E75" if "BESTANDEN" in v else ("#EF9F27" if "WARNUNG" in v else "#E24B4A")
            zone_rows += f"""
            <tr>
              <td>{zname}</td>
              <td>{zs.get('n_points', 0):,}</td>
              <td>{zs.get('tolerance_ok_mm', 0):.2f} mm</td>
              <td>{zs.get('mean_mm', 0):.3f} mm</td>
              <td>{zs.get('max_mm', 0):.3f} mm</td>
              <td>{zs.get('pct_ok', 0):.1f}%</td>
              <td style="color:{vc};font-weight:600">{v}</td>
            </tr>"""

        return f"""<!DOCTYPE html>
<html lang="de">
<head>
<meta charset="UTF-8">
<title>D405 3D-Qualitätsanalyse</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{background:#0f1117;color:#e2e2e2;font-family:system-ui,sans-serif;overflow:hidden}}
  #canvas-container{{position:absolute;top:0;left:0;width:100%;height:100%}}
  #panel{{position:absolute;top:12px;right:12px;width:320px;background:#1a1f2eee;
          border:1px solid #2a2f3e;border-radius:12px;padding:16px;overflow-y:auto;max-height:96vh}}
  h2{{font-size:14px;font-weight:500;color:#aab;margin-bottom:10px}}
  .verdict{{padding:6px 14px;border-radius:6px;font-weight:700;font-size:13px;
            display:inline-block;margin-bottom:12px;background:{verdict_color}33;
            color:{verdict_color};border:1px solid {verdict_color}66}}
  .grid{{display:grid;grid-template-columns:1fr 1fr;gap:6px;margin-bottom:12px}}
  .card{{background:#0d1b2a;border-radius:8px;padding:8px 10px}}
  .card .lbl{{font-size:10px;color:#667;text-transform:uppercase;letter-spacing:.05em}}
  .card .val{{font-size:16px;font-weight:600;margin-top:2px}}
  .card .unit{{font-size:10px;color:#667}}
  .legend{{margin-bottom:12px}}
  .leg-row{{display:flex;align-items:center;gap:8px;font-size:11px;margin-bottom:4px}}
  .dot{{width:10px;height:10px;border-radius:50%;flex-shrink:0}}
  table{{width:100%;border-collapse:collapse;font-size:10px}}
  th{{text-align:left;color:#667;padding:4px 6px;border-bottom:1px solid #2a2f3e}}
  td{{padding:4px 6px;border-bottom:1px solid #1a1f2e}}
  .tip{{font-size:10px;color:#555;margin-top:10px;line-height:1.5}}
  #info-box{{position:absolute;bottom:12px;left:12px;background:#1a1f2ecc;
             border:1px solid #2a2f3e;border-radius:8px;padding:8px 12px;
             font-size:11px;color:#aab;display:none}}
</style>
</head>
<body>
<div id="canvas-container"></div>
<div id="panel">
  <h2>D405 Qualitätsanalyse</h2>
  <div class="verdict">{overview['verdict']}</div>
  <div class="grid">
    <div class="card"><div class="lbl">Mittlere Abw.</div><div class="val">{overview['mean_mm']}</div><div class="unit">mm</div></div>
    <div class="card"><div class="lbl">RMS</div><div class="val">{overview['rms_mm']}</div><div class="unit">mm</div></div>
    <div class="card"><div class="lbl">Max. Abw.</div><div class="val">{overview['max_mm']}</div><div class="unit">mm</div></div>
    <div class="card"><div class="lbl">P95</div><div class="val">{overview['p95_mm']}</div><div class="unit">mm</div></div>
    <div class="card"><div class="lbl">OK</div><div class="val" style="color:#1D9E75">{overview['pct_ok']}%</div></div>
    <div class="card"><div class="lbl">Fehler</div><div class="val" style="color:#E24B4A">{overview['pct_fail']}%</div></div>
  </div>
  <div class="legend">
    <div class="leg-row"><div class="dot" style="background:#00C83C"></div> OK – innerhalb Toleranz</div>
    <div class="leg-row"><div class="dot" style="background:#FF9600"></div> Warnung</div>
    <div class="leg-row"><div class="dot" style="background:#E61E1E"></div> Fehler – zu viel Material</div>
    <div class="leg-row"><div class="dot" style="background:#1E1EE6"></div> Fehler – Material fehlt</div>
  </div>
  {"<h2>Zonen</h2><table><tr><th>Zone</th><th>Punkte</th><th>Tol. OK</th><th>Mittel</th><th>Max</th><th>OK%</th><th>Ergebnis</th></tr>" + zone_rows + "</table>" if zone_rows else ""}
  <div class="tip">Maus: Drehen | Scroll: Zoom | Hover: Abweichung</div>
  <div style="margin-top:8px;font-size:10px;color:#444">{overview['timestamp']} | {overview['reference']} | {overview['n_points']:,} Punkte</div>
</div>
<div id="info-box"></div>
<script>
const PTS = {_json.dumps(data)};
const scene    = new THREE.Scene();
const camera   = new THREE.PerspectiveCamera(60, window.innerWidth/window.innerHeight, 0.0001, 100);
const renderer = new THREE.WebGLRenderer({{antialias:true}});
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(window.devicePixelRatio);
document.getElementById('canvas-container').appendChild(renderer.domElement);

const n = PTS.x.length;
const geo = new THREE.BufferGeometry();
const pos = new Float32Array(n * 3);
const col = new Float32Array(n * 3);
for(let i=0;i<n;i++){{
  pos[i*3]=PTS.x[i]; pos[i*3+1]=PTS.y[i]; pos[i*3+2]=PTS.z[i];
  col[i*3]=PTS.r[i]/255; col[i*3+1]=PTS.g[i]/255; col[i*3+2]=PTS.b[i]/255;
}}
geo.setAttribute('position', new THREE.BufferAttribute(pos,3));
geo.setAttribute('color',    new THREE.BufferAttribute(col,3));
geo.computeBoundingBox();
const center = new THREE.Vector3();
geo.boundingBox.getCenter(center);
geo.translate(-center.x,-center.y,-center.z);

const mat = new THREE.PointsMaterial({{size:0.0008,vertexColors:true,sizeAttenuation:true}});
const mesh = new THREE.Points(geo,mat);
scene.add(mesh);
scene.add(new THREE.AmbientLight(0xffffff,0.5));

const size = geo.boundingBox ? Math.max(
  geo.boundingBox.max.x-geo.boundingBox.min.x,
  geo.boundingBox.max.y-geo.boundingBox.min.y,
  geo.boundingBox.max.z-geo.boundingBox.min.z) : 0.3;
camera.position.set(0, 0, size*1.8);

let isDragging=false, prevMouse={{x:0,y:0}};
let rotX=0,rotY=0;

renderer.domElement.addEventListener('mousedown', e=>{{isDragging=true;prevMouse={{x:e.clientX,y:e.clientY}}}});
renderer.domElement.addEventListener('mouseup',   ()=>isDragging=false);
renderer.domElement.addEventListener('mousemove', e=>{{
  if(!isDragging)return;
  rotY+=(e.clientX-prevMouse.x)*0.005;
  rotX+=(e.clientY-prevMouse.y)*0.005;
  prevMouse={{x:e.clientX,y:e.clientY}};
  mesh.rotation.set(rotX,rotY,0);
}});
renderer.domElement.addEventListener('wheel', e=>{{
  camera.position.z += e.deltaY*0.0002;
  camera.position.z = Math.max(0.01, Math.min(camera.position.z, size*5));
}});
window.addEventListener('resize',()=>{{
  camera.aspect=window.innerWidth/window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth,window.innerHeight);
}});

// Hover – nächsten Punkt finden
const raycaster = new THREE.Raycaster();
raycaster.params.Points.threshold = 0.002;
const mouse = new THREE.Vector2();
const infoBox = document.getElementById('info-box');
renderer.domElement.addEventListener('mousemove', e=>{{
  mouse.x=(e.clientX/window.innerWidth)*2-1;
  mouse.y=-(e.clientY/window.innerHeight)*2+1;
  raycaster.setFromCamera(mouse,camera);
  const hits = raycaster.intersectObject(mesh);
  if(hits.length>0){{
    const idx=hits[0].index;
    const dist=PTS.d[idx]; const signed=PTS.s[idx];
    const direction=signed>0?'zu viel Material':'Material fehlt';
    infoBox.style.display='block';
    infoBox.style.left=e.clientX+12+'px';
    infoBox.style.top=e.clientY-30+'px';
    infoBox.innerHTML=`Abweichung: ${{Math.abs(dist).toFixed(3)}} mm<br>${{direction}}`;
  }} else {{
    infoBox.style.display='none';
  }}
}});

function animate(){{requestAnimationFrame(animate);renderer.render(scene,camera);}}
animate();
</script>
</body>
</html>"""
