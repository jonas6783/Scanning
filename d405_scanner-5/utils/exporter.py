"""
utils/exporter.py
HTML-Report-Generator und Datei-Export für Scan-Ergebnisse.
"""

from pathlib import Path
from typing import Optional, Dict
from datetime import datetime
import json
import base64

from utils.logger import setup_logger

log = setup_logger("exporter")

try:
    import open3d as o3d
    O3D_AVAILABLE = True
except ImportError:
    O3D_AVAILABLE = False


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="de">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<style>
  :root {{
    --ok:   #1D9E75;
    --warn: #EF9F27;
    --fail: #E24B4A;
    --bg:   #0f1117;
    --card: #1a1f2e;
    --text: #e2e2e2;
    --muted:#8a8a9a;
    --border:#2a2f3e;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: system-ui, sans-serif; background: var(--bg); color: var(--text);
          line-height: 1.6; padding: 2rem; }}
  h1 {{ font-size: 1.6rem; font-weight: 600; margin-bottom: 0.25rem; }}
  h2 {{ font-size: 1.1rem; font-weight: 500; margin: 1.5rem 0 0.75rem; color: #aab; }}
  .header {{ border-bottom: 1px solid var(--border); padding-bottom: 1rem; margin-bottom: 2rem; }}
  .meta {{ color: var(--muted); font-size: 0.85rem; }}
  .verdict {{
    display: inline-block; padding: 0.4rem 1.2rem; border-radius: 6px;
    font-weight: 600; font-size: 1rem; margin-top: 0.5rem;
    background: {verdict_bg}; color: white;
  }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
           gap: 1rem; margin: 1rem 0; }}
  .card {{ background: var(--card); border: 1px solid var(--border); border-radius: 10px;
           padding: 1rem 1.25rem; }}
  .card .label {{ font-size: 0.75rem; color: var(--muted); text-transform: uppercase;
                  letter-spacing: 0.05em; }}
  .card .value {{ font-size: 1.4rem; font-weight: 600; margin-top: 0.2rem; }}
  .card .unit  {{ font-size: 0.75rem; color: var(--muted); }}
  .bar-row {{ display: flex; height: 22px; border-radius: 6px; overflow: hidden;
              margin: 0.75rem 0; }}
  .bar-ok   {{ background: var(--ok);   }}
  .bar-warn {{ background: var(--warn); }}
  .bar-fail {{ background: var(--fail); }}
  .bar-legend {{ display: flex; gap: 1.5rem; font-size: 0.8rem; margin-bottom: 1rem; }}
  .dot {{ width: 10px; height: 10px; border-radius: 50%; display: inline-block;
          margin-right: 5px; vertical-align: middle; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.88rem; }}
  th {{ text-align: left; color: var(--muted); font-weight: 500; padding: 0.4rem 0.75rem;
        border-bottom: 1px solid var(--border); }}
  td {{ padding: 0.45rem 0.75rem; border-bottom: 1px solid var(--border); }}
  tr:last-child td {{ border-bottom: none; }}
  .img-wrap {{ margin: 1.5rem 0; }}
  .img-wrap img {{ width: 100%; max-width: 900px; border-radius: 10px;
                   border: 1px solid var(--border); }}
  .footer {{ margin-top: 3rem; padding-top: 1rem; border-top: 1px solid var(--border);
             font-size: 0.78rem; color: var(--muted); }}
  .tag {{ display: inline-block; padding: 0.15rem 0.5rem; border-radius: 4px;
          font-size: 0.75rem; font-weight: 600; }}
  .tag-ok   {{ background: rgba(29,158,117,0.2); color: var(--ok);   }}
  .tag-warn {{ background: rgba(239,159,39,0.2);  color: var(--warn); }}
  .tag-fail {{ background: rgba(226,75,74,0.2);   color: var(--fail); }}
</style>
</head>
<body>

<div class="header">
  <h1>{title}</h1>
  <p class="meta">Erstellt: {timestamp} &nbsp;|&nbsp; Kamera: Intel RealSense D405
    &nbsp;|&nbsp; Referenz: {reference_file}</p>
  <div class="verdict">{verdict}</div>
</div>

<h2>Übersicht</h2>
<div class="grid">
  <div class="card">
    <div class="label">Mittlere Abweichung</div>
    <div class="value">{mean_mm:.3f}</div>
    <div class="unit">mm</div>
  </div>
  <div class="card">
    <div class="label">RMS-Abweichung</div>
    <div class="value">{rms_mm:.3f}</div>
    <div class="unit">mm</div>
  </div>
  <div class="card">
    <div class="label">Max. Abweichung</div>
    <div class="value">{max_mm:.3f}</div>
    <div class="unit">mm</div>
  </div>
  <div class="card">
    <div class="label">95. Perzentil</div>
    <div class="value">{p95_mm:.3f}</div>
    <div class="unit">mm</div>
  </div>
  <div class="card">
    <div class="label">Analysierte Punkte</div>
    <div class="value">{n_points_k}</div>
    <div class="unit">× 1000</div>
  </div>
  <div class="card">
    <div class="label">Standardabw.</div>
    <div class="value">{std_mm:.3f}</div>
    <div class="unit">mm</div>
  </div>
</div>

<h2>Toleranz-Auswertung</h2>
<div class="bar-legend">
  <span><span class="dot" style="background:var(--ok)"></span>
    OK ≤{tolerance_ok_mm:.1f} mm ({pct_ok:.1f}%)</span>
  <span><span class="dot" style="background:var(--warn)"></span>
    Warnung ≤{tolerance_warn_mm:.1f} mm ({pct_warn:.1f}%)</span>
  <span><span class="dot" style="background:var(--fail)"></span>
    Fehler &gt;{tolerance_warn_mm:.1f} mm ({pct_fail:.1f}%)</span>
</div>
<div class="bar-row">
  <div class="bar-ok"   style="width:{pct_ok:.1f}%"></div>
  <div class="bar-warn" style="width:{pct_warn:.1f}%"></div>
  <div class="bar-fail" style="width:{pct_fail:.1f}%"></div>
</div>

<h2>Detailstatistiken</h2>
<table>
  <tr><th>Kennwert</th><th>Wert</th><th>Bewertung</th></tr>
  <tr><td>Mittlere Abweichung</td>
      <td>{mean_mm:.4f} mm</td>
      <td><span class="tag {tag_mean}">{mean_mm:.4f} mm</span></td></tr>
  <tr><td>RMS-Abweichung</td>
      <td>{rms_mm:.4f} mm</td>
      <td><span class="tag {tag_rms}">{rms_mm:.4f} mm</span></td></tr>
  <tr><td>Median-Abweichung</td>
      <td>{median_mm:.4f} mm</td>
      <td>–</td></tr>
  <tr><td>Standardabweichung</td>
      <td>{std_mm:.4f} mm</td>
      <td>–</td></tr>
  <tr><td>Maximale Abweichung</td>
      <td>{max_mm:.4f} mm</td>
      <td><span class="tag {tag_max}">{max_mm:.4f} mm</span></td></tr>
  <tr><td>99. Perzentil</td>
      <td>{p99_mm:.4f} mm</td>
      <td>–</td></tr>
  <tr><td>Material zu viel (max +)</td>
      <td>{max_positive_mm:.4f} mm</td>
      <td>–</td></tr>
  <tr><td>Material fehlt (max −)</td>
      <td>{max_negative_mm:.4f} mm</td>
      <td>–</td></tr>
  <tr><td>Punkte in Toleranz (OK)</td>
      <td>{n_ok:,}</td>
      <td>{pct_ok:.2f}%</td></tr>
  <tr><td>Punkte in Warnung</td>
      <td>{n_warn:,}</td>
      <td>{pct_warn:.2f}%</td></tr>
  <tr><td>Punkte außerhalb</td>
      <td>{n_fail:,}</td>
      <td>{pct_fail:.2f}%</td></tr>
</table>

{heatmap_section}

<div class="footer">
  D405 Qualitätskontrolle-System &nbsp;|&nbsp;
  Toleranzen: OK ≤{tolerance_ok_mm:.1f} mm / WARN ≤{tolerance_warn_mm:.1f} mm &nbsp;|&nbsp;
  Erstellt mit Python + Open3D &nbsp;|&nbsp; {timestamp}
</div>

</body>
</html>
"""


class ReportExporter:
    """Erstellt HTML-Reports und exportiert Scan-Ergebnisse."""

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_html_report(
        self,
        stats: Dict,
        heatmap_path: Optional[Path] = None,
        title: str = "D405 Qualitätskontrolle"
    ) -> Path:
        """Generiert einen vollständigen HTML-Report."""

        # Toleranz-Tags bestimmen
        def tag_class(val_mm, ok_mm, warn_mm):
            if val_mm <= ok_mm:   return "tag-ok"
            if val_mm <= warn_mm: return "tag-warn"
            return "tag-fail"

        ok_mm   = stats.get("tolerance_ok_mm",   0.5)
        warn_mm = stats.get("tolerance_warn_mm",  1.0)

        verdict = stats.get("verdict", "–")
        verdict_colors = {
            "BESTANDEN":        "#1D9E75",
            "WARNUNG":          "#EF9F27",
            "NICHT BESTANDEN":  "#E24B4A",
        }
        verdict_bg = verdict_colors.get(verdict, "#444")

        # Heatmap als Base64 einbetten
        heatmap_section = ""
        if heatmap_path and Path(heatmap_path).exists():
            with open(heatmap_path, "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode()
            heatmap_section = f"""
<h2>Abstandsverteilung (Heatmap)</h2>
<div class="img-wrap">
  <img src="data:image/png;base64,{img_b64}" alt="Heatmap">
</div>"""

        html = HTML_TEMPLATE.format(
            title=title,
            timestamp=stats.get("timestamp", datetime.now().isoformat())[:19].replace("T", " "),
            reference_file=Path(stats.get("reference_file", "–")).name,
            verdict=verdict,
            verdict_bg=verdict_bg,
            mean_mm=stats.get("mean_mm", 0),
            rms_mm=stats.get("rms_mm", 0),
            max_mm=stats.get("max_mm", 0),
            p95_mm=stats.get("p95_mm", 0),
            p99_mm=stats.get("p99_mm", 0),
            std_mm=stats.get("std_mm", 0),
            median_mm=stats.get("median_mm", 0),
            max_positive_mm=stats.get("max_positive_mm", 0),
            max_negative_mm=stats.get("max_negative_mm", 0),
            n_points_k=f"{stats.get('n_points_analyzed', 0) // 1000:.0f}",
            n_ok=stats.get("n_ok", 0),
            n_warn=stats.get("n_warn", 0),
            n_fail=stats.get("n_fail", 0),
            pct_ok=stats.get("pct_ok", 0),
            pct_warn=stats.get("pct_warn", 0),
            pct_fail=stats.get("pct_fail", 0),
            tolerance_ok_mm=ok_mm,
            tolerance_warn_mm=warn_mm,
            tag_mean=tag_class(stats.get("mean_mm", 0), ok_mm, warn_mm),
            tag_rms=tag_class(stats.get("rms_mm", 0), ok_mm, warn_mm),
            tag_max=tag_class(stats.get("max_mm", 0), ok_mm, warn_mm),
            heatmap_section=heatmap_section,
        )

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.output_dir / f"qk_report_{timestamp}.html"
        report_path.write_text(html, encoding="utf-8")
        log.info(f"HTML-Report gespeichert: {report_path}")
        return report_path

    def save_stats_json(self, stats: Dict, path: Optional[Path] = None) -> Path:
        """Speichert Statistiken als JSON."""
        if path is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = self.output_dir / f"stats_{ts}.json"

        # Numpy-Arrays und nicht-serialisierbare Objekte entfernen
        import numpy as _np
        clean = {}
        for k, v in stats.items():
            if isinstance(v, (str, bool)):
                clean[k] = v
            elif isinstance(v, int):
                clean[k] = v
            elif isinstance(v, float):
                clean[k] = v
            elif isinstance(v, _np.integer):
                clean[k] = int(v)
            elif isinstance(v, _np.floating):
                clean[k] = float(v)
            elif isinstance(v, dict):
                # Zonen-Stats rekursiv sauber machen
                sub = {}
                for sk, sv in v.items():
                    if isinstance(sv, (str, bool, int, float)):
                        sub[sk] = sv
                    elif isinstance(sv, _np.integer):
                        sub[sk] = int(sv)
                    elif isinstance(sv, _np.floating):
                        sub[sk] = float(sv)
                clean[k] = sub
            # numpy arrays, PointCloud, ndarray werden übersprungen

        path = Path(path)
        path.write_text(json.dumps(clean, indent=2, ensure_ascii=False), encoding="utf-8")
        log.info(f"Statistiken gespeichert: {path}")
        return path
