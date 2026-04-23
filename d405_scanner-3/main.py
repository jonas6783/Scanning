"""
main.py
Haupteinstieg – kompletter D405 Scan-Workflow für Gussbauteile.

Verwendung:
    python main.py --mode single
    python main.py --mode turntable --positions 12
    python main.py --mode compare --reference teil.stl
    python main.py --mode full --reference teil.stl --positions 12
"""

import argparse
import sys
import time
from pathlib import Path
from datetime import datetime

from config.settings import ScannerConfig, DEFAULT_CONFIG
from core.camera import D405Camera
from core.capture import FrameCapture
from processing.pointcloud import PointCloudProcessor
from processing.mesh import MeshReconstructor
from analysis.quality import QualityAnalyzer
from utils.logger import setup_logger
from utils.exporter import ReportExporter

try:
    import open3d as o3d
    O3D_AVAILABLE = True
except ImportError:
    O3D_AVAILABLE = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="D405 Gussbauteil-Scanner – Professionelle Qualitätskontrolle",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Beispiele:
  python main.py --mode single
  python main.py --mode turntable --positions 12
  python main.py --mode compare --reference mein_teil.stl
  python main.py --mode full --reference mein_teil.stl --positions 12 --output scan_output
        """
    )
    parser.add_argument(
        "--mode", choices=["single", "turntable", "compare", "full"],
        default="full",
        help="Scan-Modus (default: full)"
    )
    parser.add_argument(
        "--reference", type=Path, default=None,
        help="Pfad zur CAD-Referenzdatei (.stl / .ply / .obj)"
    )
    parser.add_argument(
        "--positions", type=int, default=None,
        help="Anzahl Drehteller-Positionen (überschreibt Konfiguration)"
    )
    parser.add_argument(
        "--scan", type=Path, default=None,
        help="Vorhandene Scan-Datei für Vergleich (überspringt Aufnahme)"
    )
    parser.add_argument(
        "--output", type=Path, default=Path("output"),
        help="Ausgabeverzeichnis (default: output/)"
    )
    parser.add_argument(
        "--no-mesh", action="store_true",
        help="Mesh-Rekonstruktion überspringen"
    )
    parser.add_argument(
        "--no-report", action="store_true",
        help="HTML-Report nicht erstellen"
    )
    parser.add_argument(
        "--laser-power", type=int, default=None,
        help="Laser-Power 0–360 (default: 150)"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Ausführliche Debug-Ausgabe"
    )
    return parser.parse_args()


# ──────────────────────────────────────────────────────────────────────
# Workflow-Schritte
# ──────────────────────────────────────────────────────────────────────

def step_capture(cfg: ScannerConfig, args: argparse.Namespace, log) -> list:
    """Schritt 1: Kamera initialisieren und Scans aufnehmen."""
    log.info("=" * 60)
    log.info("SCHRITT 1: AUFNAHME")
    log.info("=" * 60)

    # Parameter überschreiben wenn angegeben
    if args.positions:
        cfg.capture.turntable_positions = args.positions
    if args.laser_power:
        cfg.camera.laser_power = args.laser_power

    camera = D405Camera(cfg.camera, cfg.filters)

    if not camera.start():
        log.error("Kamera konnte nicht gestartet werden!")
        sys.exit(1)

    log.info(f"Kamera gestartet – Modus: {args.mode}")
    log.info("WICHTIG: AESUB Orange Spray auf das Bauteil aufbringen!")
    log.info("         Trockenzeit: ~2 Minuten")
    input("\nBauteil vorbereitet? ENTER zum Fortfahren...")

    capture = FrameCapture(camera, cfg.capture, cfg.camera)

    # Kamera einpendeln lassen
    camera.warmup(frames=cfg.capture.warmup_frames)

    point_clouds = []

    try:
        if args.mode in ("single", "compare"):
            # Einzelaufnahme
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            pcd = capture.capture_single(label=f"scan_{ts}")
            if pcd:
                point_clouds = [pcd]
                save_path = cfg.paths.scans_dir / f"scan_{ts}.ply"
                if O3D_AVAILABLE:
                    o3d.io.write_point_cloud(str(save_path), pcd)
                    log.info(f"Scan gespeichert: {save_path}")

        elif args.mode in ("turntable", "full"):
            # Drehteller-Scan
            log.info(f"Drehteller-Scan: {cfg.capture.turntable_positions} Positionen")
            log.info("Drehteller bereitstellen oder manuelle Positionierung bestätigen")
            point_clouds = capture.capture_turntable(
                output_dir=cfg.paths.scans_dir
            )

    finally:
        camera.stop()

    if not point_clouds:
        log.error("Keine Punktwolken aufgenommen!")
        sys.exit(1)

    log.info(f"Aufnahme abgeschlossen: {len(point_clouds)} Scans")
    return point_clouds


def step_process(
    cfg: ScannerConfig,
    point_clouds: list,
    log
) -> object:
    """Schritt 2: Vorverarbeitung, Registrierung, Merge."""
    log.info("=" * 60)
    log.info("SCHRITT 2: VERARBEITUNG")
    log.info("=" * 60)

    processor = PointCloudProcessor(cfg.processing)

    # Vorverarbeitung jeder Einzelpunktwolke
    processed = []
    for i, pcd in enumerate(point_clouds):
        log.info(f"Verarbeite Scan {i+1}/{len(point_clouds)}...")
        clean = processor.preprocess(pcd, label=f"scan_{i+1}")
        if clean and len(clean.points) > 0:
            processed.append(clean)

    if not processed:
        log.error("Keine Punktwolken nach Vorverarbeitung!")
        sys.exit(1)

    # Registrierung und Merge
    if len(processed) > 1:
        log.info("Starte ICP-Registrierung...")
        final_pcd = processor.register_multiple(processed)
    else:
        final_pcd = processed[0]

    # Statistiken ausgeben
    stats = processor.compute_stats(final_pcd)

    # Gesamtpunktwolke speichern
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    pcd_path = cfg.paths.scans_dir / f"final_scan_{ts}.ply"
    if O3D_AVAILABLE and final_pcd:
        o3d.io.write_point_cloud(str(pcd_path), final_pcd)
        log.info(f"Finale Punktwolke gespeichert: {pcd_path}")

    return final_pcd


def step_mesh(
    cfg: ScannerConfig,
    pcd: object,
    log
) -> object:
    """Schritt 3: Mesh-Rekonstruktion."""
    log.info("=" * 60)
    log.info("SCHRITT 3: MESH-REKONSTRUKTION")
    log.info("=" * 60)

    reconstructor = MeshReconstructor(cfg.processing)
    mesh = reconstructor.build_mesh(pcd)

    if mesh is None:
        log.error("Mesh-Rekonstruktion fehlgeschlagen!")
        return None

    # Mesh speichern
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    mesh_base = cfg.paths.meshes_dir / f"mesh_{ts}"
    reconstructor.save(mesh, mesh_base, fmt="stl")
    reconstructor.save(mesh, mesh_base, fmt="ply")

    return mesh


def step_analysis(
    cfg: ScannerConfig,
    pcd: object,
    reference_path: Path,
    args: argparse.Namespace,
    log
) -> dict:
    """Schritt 4: Qualitätsanalyse und Report."""
    log.info("=" * 60)
    log.info("SCHRITT 4: QUALITÄTSANALYSE")
    log.info("=" * 60)

    analyzer = QualityAnalyzer(cfg.analysis)
    exporter = ReportExporter(cfg.paths.reports_dir)

    # CAD-Vergleich
    stats = analyzer.compare_to_reference(pcd, reference_path)

    if stats is None:
        log.error("Qualitätsanalyse fehlgeschlagen!")
        return {}

    # Heatmap speichern
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    heatmap_path = cfg.paths.reports_dir / f"heatmap_{ts}.png"
    analyzer.save_heatmap(stats, heatmap_path)

    # Farbige Punktwolke speichern
    if O3D_AVAILABLE and "colored_pcd" in stats:
        colored_path = cfg.paths.scans_dir / f"scan_colored_{ts}.ply"
        o3d.io.write_point_cloud(str(colored_path), stats["colored_pcd"])
        log.info(f"Farbige Abweichungs-Punktwolke: {colored_path}")

    # JSON-Statistiken
    exporter.save_stats_json(stats)

    # HTML-Report
    if not args.no_report:
        report_path = exporter.generate_html_report(
            stats,
            heatmap_path=heatmap_path,
            title=cfg.analysis.report_title
        )
        log.info(f"HTML-Report: {report_path}")
        log.info(f"  → Öffnen: file://{report_path.resolve()}")

    return stats


# ──────────────────────────────────────────────────────────────────────
# Hauptprogramm
# ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Konfiguration
    cfg = DEFAULT_CONFIG
    cfg.paths.output_dir  = args.output
    cfg.paths.scans_dir   = args.output / "scans"
    cfg.paths.meshes_dir  = args.output / "meshes"
    cfg.paths.reports_dir = args.output / "reports"
    cfg.paths.create_all()

    # Logger
    log = setup_logger("main", log_dir=args.output)

    log.info("╔══════════════════════════════════════════════════════╗")
    log.info("║    D405 Gussbauteil-Scanner – Qualitätskontrolle     ║")
    log.info("╚══════════════════════════════════════════════════════╝")
    log.info(f"Modus: {args.mode}")
    log.info(f"Ausgabe: {args.output.resolve()}")

    t_start = time.time()
    final_pcd = None

    # ── Vorhandenen Scan laden ──────────────────────────────────────
    if args.scan:
        if not args.scan.exists():
            log.error(f"Scan-Datei nicht gefunden: {args.scan}")
            sys.exit(1)
        log.info(f"Lade vorhandenen Scan: {args.scan}")
        if O3D_AVAILABLE:
            final_pcd = o3d.io.read_point_cloud(str(args.scan))
            log.info(f"Scan geladen: {len(final_pcd.points):,} Punkte")
        else:
            log.error("open3d nicht verfügbar – kann Scan nicht laden")
            sys.exit(1)

    # ── Neue Aufnahme ──────────────────────────────────────────────
    if final_pcd is None and args.mode in ("single", "turntable", "full"):
        point_clouds = step_capture(cfg, args, log)
        final_pcd = step_process(cfg, point_clouds, log)

    if final_pcd is None:
        log.error("Keine Punktwolke vorhanden!")
        sys.exit(1)

    # ── Mesh-Rekonstruktion ────────────────────────────────────────
    if not args.no_mesh:
        step_mesh(cfg, final_pcd, log)

    # ── Qualitätsanalyse ───────────────────────────────────────────
    if args.reference:
        if not args.reference.exists():
            log.error(f"Referenz-Datei nicht gefunden: {args.reference}")
            sys.exit(1)
        stats = step_analysis(cfg, final_pcd, args.reference, args, log)

        # Ergebnis-Zusammenfassung
        t_total = time.time() - t_start
        log.info("=" * 60)
        log.info(f"FERTIG in {t_total:.1f} Sekunden")
        log.info(f"Ergebnis: {stats.get('verdict', '–')}")
        log.info(f"Mittlere Abweichung: {stats.get('mean_mm', 0):.3f} mm")
        log.info(f"RMS-Abweichung:      {stats.get('rms_mm', 0):.3f} mm")
        log.info("=" * 60)
    else:
        t_total = time.time() - t_start
        log.info(f"Fertig in {t_total:.1f} s – Scan gespeichert in: {args.output.resolve()}")
        if args.mode != "compare":
            log.info("Tipp: --reference mein_bauteil.stl angeben für Qualitätsanalyse")


if __name__ == "__main__":
    main()
