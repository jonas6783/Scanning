"""
CAD-Vergleich: Scan-Punktwolke vs. CAD-Modell.

Pipeline:
  1. Lade Scan (.ply) und CAD (.stl)
  2. Einheiten automatisch vereinheitlichen
  3. Scan saeubern (DBSCAN, groesstes Cluster)
  4. CAD zu Punktwolke samplen (Poisson Disk)
  5. Globale Registrierung mit FPFH + RANSAC
  6. Hidden Point Removal: nur sichtbare CAD-Seite
  7. Multi-Scale ICP (5mm -> 1mm -> 0.3mm) mit Tukey-Loss
  8. Vorzeichenbehaftete Distanzen Scan -> CAD
  9. Defekt-Cluster mit DBSCAN
 10. HTML-Report mit interaktiver 3D-Heatmap

Voraussetzungen:
  pip install open3d plotly numpy

Nutzung:
  python cad_compare.py pointclouds/pointcloud_000.ply bauteil.stl
  python cad_compare.py scan.ply bauteil.stl --tolerance-mm 0.5
"""

import argparse
import copy
import os
import sys
from datetime import datetime

import numpy as np
import open3d as o3d
import plotly.graph_objects as go

# --- KONFIGURATION ---------------------------------------------------------
DEFAULT_TOLERANCE_MM = 1.0

CAD_SAMPLE_POINTS    = 100_000   # Anzahl Punkte fuers CAD-Sampling

VOXEL_COARSE         = 0.005     # 5 mm  (Globale Reg + ICP grob)
VOXEL_MEDIUM         = 0.001     # 1 mm  (ICP mittel)
VOXEL_FINE           = 0.0003    # 0.3 mm (ICP fein)

SCAN_DBSCAN_EPS      = 0.002     # 2 mm Cluster-Abstand fuer Scan-Cleanup
SCAN_DBSCAN_MIN_PTS  = 20

DEFECT_DBSCAN_EPS    = 0.003     # 3 mm Cluster-Abstand fuer Defekt-Suche
DEFECT_MIN_POINTS    = 30        # Cluster < diese Groesse werden ignoriert

VIZ_MAX_POINTS       = 100_000   # ueber dem Wert wird fuer den Browser gesampled

OUTPUT_DIR = "reports"

# --- HILFSFUNKTIONEN -------------------------------------------------------
def step(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def detect_units_and_scale(mesh, scan_pcd, manual_scale=None):
    """Bringt das Mesh in dieselbe Einheit wie der Scan (Meter).
    Nutzt 5-95% Perzentil-BBox, die ist gegen Ausreisser robust."""
    mesh_pts = np.asarray(mesh.vertices)
    scan_pts = np.asarray(scan_pcd.points)

    mesh_raw   = np.asarray(mesh.get_max_bound() - mesh.get_min_bound())
    scan_raw   = np.asarray(scan_pcd.get_max_bound() - scan_pcd.get_min_bound())
    mesh_robust = robust_extent(mesh_pts)
    scan_robust = robust_extent(scan_pts)

    step(f"   Mesh BBox roh:    {mesh_raw[0]:.2f} x {mesh_raw[1]:.2f} x {mesh_raw[2]:.2f}")
    step(f"   Mesh BBox 5-95%:  {mesh_robust[0]:.2f} x {mesh_robust[1]:.2f} x {mesh_robust[2]:.2f}")
    step(f"   Scan BBox roh:    {scan_raw[0]:.4f} x {scan_raw[1]:.4f} x {scan_raw[2]:.4f}  m")
    step(f"   Scan BBox 5-95%:  {scan_robust[0]:.4f} x {scan_robust[1]:.4f} x {scan_robust[2]:.4f}  m")

    if scan_raw.max() > scan_robust.max() * 3:
        step(f"   Hinweis: Scan hat Outlier (BBox roh {scan_raw.max():.3f} m vs robust {scan_robust.max():.3f} m)")

    if manual_scale is not None:
        step(f"   Manuelle Skalierung: x {manual_scale}")
        mesh.scale(manual_scale, center=(0, 0, 0))
        new_ext = robust_extent(np.asarray(mesh.vertices))
        step(f"   Mesh nach Skalierung (5-95%): "
             f"{new_ext[0]:.4f} x {new_ext[1]:.4f} x {new_ext[2]:.4f}  m")
        return mesh

    # Robustes Verhaeltnis fuer die Auto-Erkennung
    ratio = mesh_robust.max() / max(scan_robust.max(), 1e-6)
    step(f"   Verhaeltnis robust: {ratio:.1f}")

    candidates = [(0.001, "mm -> m"), (0.01, "cm -> m"),
                  (0.0254, "inch -> m"), (1.0, "keine")]
    for factor, label in candidates:
        new_ratio = ratio * factor
        # Engerer Bereich: nach Skalierung sollten Mesh und Scan etwa gleich gross sein
        if 0.5 <= new_ratio <= 2.0:
            if factor != 1.0:
                step(f"   Auto-Skalierung: {label} (x {factor})")
                mesh.scale(factor, center=(0, 0, 0))
            else:
                step("   Keine Skalierung noetig")
            return mesh

    print()
    print("FEHLER: Skalierung kann nicht automatisch erkannt werden.")
    print(f"  Robustes Verhaeltnis {ratio:.1f} passt zu keiner Standard-Einheit.")
    print("  Manueller Override:")
    print("    --cad-scale 0.001   # CAD in mm")
    print("    --cad-scale 0.01    # CAD in cm")
    print("    --cad-scale 1.0     # CAD bereits in m")
    sys.exit(1)


def clean_scan(pcd):
    """1) DBSCAN -> nur groesstes Cluster (entfernt Tisch / Hintergrund-Cluster)
       2) Statistical Outlier Removal (entfernt schwebende Einzel-Ausreisser)."""
    if len(pcd.points) < 100:
        return pcd
    labels = np.array(pcd.cluster_dbscan(
        eps=SCAN_DBSCAN_EPS, min_points=SCAN_DBSCAN_MIN_PTS, print_progress=False))
    if labels.max() >= 0:
        largest = np.bincount(labels[labels >= 0]).argmax()
        keep = np.where(labels == largest)[0]
        pcd = pcd.select_by_index(keep)
    # SOR gegen Outlier (Reflektionen, Maskenraender, etc.)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    return pcd


def robust_extent(points_np):
    """5-95% Perzentil-BBox - resistent gegen einzelne Ausreisser."""
    if len(points_np) == 0:
        return np.zeros(3)
    return np.percentile(points_np, 95, axis=0) - np.percentile(points_np, 5, axis=0)


def sample_cad(mesh, n_points):
    """CAD-Mesh zu gleichmaessiger Punktwolke.
    Uniform sampling ist robuster und schneller als Poisson Disk bei grossen Meshes."""
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()
    if not mesh.has_triangle_normals():
        mesh.compute_triangle_normals()
    pcd = mesh.sample_points_uniformly(number_of_points=n_points, use_triangle_normal=True)
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.005, max_nn=30))
    return pcd


def pca_axes(points_np):
    """PCA-Hauptachsen + Schwerpunkt einer Punktwolke."""
    centroid = points_np.mean(axis=0)
    centered = points_np - centroid
    cov = (centered.T @ centered) / max(len(centered) - 1, 1)
    eigval, eigvec = np.linalg.eigh(cov)
    order = np.argsort(eigval)[::-1]
    return centroid, eigvec[:, order]


def all_pca_rotation_hypotheses(scan_pts, cad_pts):
    """24 Rotations-Hypothesen: alle 6 Permutationen der Hauptachsen
    x 4 gueltige Vorzeichen-Kombinationen (Determinante = +1).
    Deckt 90/180/270 Grad Rotationen um Hauptachsen ab."""
    import itertools
    s_c, s_axes = pca_axes(scan_pts)
    c_c, c_axes = pca_axes(cad_pts)

    poses = []
    for perm in itertools.permutations([0, 1, 2]):
        for signs in itertools.product([1, -1], repeat=3):
            c_mod = c_axes[:, list(perm)] * np.array(signs)
            if np.linalg.det(c_mod) < 0:
                continue  # Spiegelung -> keine Rotation
            R = s_axes @ c_mod.T
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = s_c - R @ c_c
            poses.append(T)
    return poses


def global_registration(cad_pcd, scan_pcd, voxel):
    """24 PCA-Hypothesen grob durchtesten, Top 3 fein verfeinern.
    Deckt 90/180/270 Grad Symmetrien ab, die einfache 4-Hypothesen-PCA verfehlt."""
    cad_pts = np.asarray(cad_pcd.points)
    scan_pts = np.asarray(scan_pcd.points)

    cad_down = cad_pcd.voxel_down_sample(voxel)
    scan_down = scan_pcd.voxel_down_sample(voxel)
    cad_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 2, max_nn=30))
    scan_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 2, max_nn=30))

    poses = all_pca_rotation_hypotheses(scan_pts, cad_pts)
    step(f"   Pruefe {len(poses)} PCA-Hypothesen...")

    p2plane = o3d.pipelines.registration.TransformationEstimationPointToPlane()
    quick = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=20)

    # Stufe 1: schnelle Grob-ICP fuer alle 24
    rough = []
    for T_init in poses:
        res = o3d.pipelines.registration.registration_icp(
            cad_down, scan_down, voxel * 15, T_init, p2plane, quick)
        rough.append((res.fitness, res.transformation))
    rough.sort(key=lambda x: -x[0])

    step(f"   Top 3 Fitness aus {len(poses)}: "
         f"{rough[0][0]:.3f}, {rough[1][0]:.3f}, {rough[2][0]:.3f}")

    # Stufe 2: nur Top 3 fein verfeinern (shrinking distance)
    fine = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=30)
    best = None
    for rank, (_, T_init) in enumerate(rough[:3]):
        T = T_init
        for max_d in (voxel * 10, voxel * 4, voxel * 2):
            res = o3d.pipelines.registration.registration_icp(
                cad_down, scan_down, max_d, T, p2plane, fine)
            T = res.transformation
        step(f"   Kandidat {rank+1}: fitness {res.fitness:.3f} | RMSE {res.inlier_rmse*1000:.2f} mm")
        if best is None or res.fitness > best.fitness:
            best = res
    return best


def hidden_point_removal(pcd, camera_pos):
    """Nur Punkte zurueckgeben, die von der Kamera-Position aus sichtbar waeren."""
    diameter = np.linalg.norm(np.asarray(pcd.get_max_bound()) - np.asarray(pcd.get_min_bound()))
    radius = diameter * 100
    _, idx = pcd.hidden_point_removal(camera_pos, radius)
    return pcd.select_by_index(idx)


def refine_icp(source, target, init_T, voxel, max_iter=50):
    """Multi-scale ICP mit Tukey-Loss + shrinking correspondence distance.
    Die shrinking-Logik frisst grosse Initial-Offsets - faengt mit voxel*10 an."""
    src_d = source.voxel_down_sample(voxel)
    tgt_d = target.voxel_down_sample(voxel)
    src_d.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 2, max_nn=30))
    tgt_d.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 2, max_nn=30))
    loss = o3d.pipelines.registration.TukeyLoss(k=voxel * 2)

    T = init_T
    for max_d in (voxel * 10, voxel * 4, voxel * 2):
        result = o3d.pipelines.registration.registration_icp(
            src_d, tgt_d, max_d, T,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(loss),
            o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=max(max_iter // 3, 15)))
        T = result.transformation
    return result


def signed_distances(points_np, mesh):
    """Vorzeichenbehaftete Distanz: + = ausserhalb (Burr), - = innerhalb (fehlend)."""
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    query = o3d.core.Tensor(points_np.astype(np.float32))
    return scene.compute_signed_distance(query).numpy()


def find_defect_clusters(points, signed_dists, tolerance):
    """DBSCAN auf den Punkten ausserhalb der Toleranz."""
    mask = np.abs(signed_dists) > tolerance
    defect_pts = points[mask]
    defect_d = signed_dists[mask]

    if len(defect_pts) < DEFECT_MIN_POINTS:
        return [], defect_pts, defect_d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(defect_pts)
    labels = np.array(pcd.cluster_dbscan(
        eps=DEFECT_DBSCAN_EPS, min_points=DEFECT_MIN_POINTS, print_progress=False))

    clusters = []
    for lab in range(labels.max() + 1):
        idx = labels == lab
        if idx.sum() < DEFECT_MIN_POINTS:
            continue
        cd = defect_d[idx]
        clusters.append({
            'id': len(clusters) + 1,
            'n_points': int(idx.sum()),
            'center': defect_pts[idx].mean(axis=0),
            'max_dev_mm': float(np.abs(cd).max() * 1000),
            'mean_dev_mm': float(np.abs(cd).mean() * 1000),
            'type': 'ueberschuessig' if cd.mean() > 0 else 'fehlend',
        })
    clusters.sort(key=lambda c: c['max_dev_mm'], reverse=True)
    for i, c in enumerate(clusters):
        c['id'] = i + 1
    return clusters, defect_pts, defect_d


def make_html_report(scan_pts, signed_dists, clusters, tolerance,
                     output_path, scan_path, cad_path, fitness, rmse,
                     mesh_aligned=None):
    abs_d_mm = np.abs(signed_dists) * 1000
    n_total = len(scan_pts)
    n_def = int((abs_d_mm > tolerance * 1000).sum())
    pass_fail = "PASS" if not clusters else "FAIL"
    pass_color = "#10b981" if pass_fail == "PASS" else "#ef4444"

    # Visualisierung kann bei sehr vielen Punkten den Browser ausbremsen -> samplen
    if n_total > VIZ_MAX_POINTS:
        sel = np.random.choice(n_total, VIZ_MAX_POINTS, replace=False)
        viz_pts = scan_pts[sel]
        viz_d = signed_dists[sel]
    else:
        viz_pts = scan_pts
        viz_d = signed_dists

    color_mm = viz_d * 1000
    cmax = max(np.abs(color_mm).max(), tolerance * 1000 * 2)

    fig = go.Figure()

    # CAD-Mesh als graue Punkte (toggelbar via Legende)
    if mesh_aligned is not None and len(mesh_aligned.vertices) > 0:
        cad_sample = mesh_aligned.sample_points_uniformly(number_of_points=20000)
        cad_pts_mm = np.asarray(cad_sample.points) * 1000
        fig.add_trace(go.Scatter3d(
            x=cad_pts_mm[:, 0], y=cad_pts_mm[:, 1], z=cad_pts_mm[:, 2],
            mode='markers',
            marker=dict(size=1.5, color='#888', opacity=0.25),
            name='CAD-Modell', hoverinfo='skip',
        ))

    # Scan mit Heatmap-Farben
    fig.add_trace(go.Scatter3d(
        x=viz_pts[:, 0] * 1000, y=viz_pts[:, 1] * 1000, z=viz_pts[:, 2] * 1000,
        mode='markers',
        marker=dict(
            size=2, color=color_mm, colorscale='RdBu_r',
            cmin=-cmax, cmax=cmax,
            colorbar=dict(title=dict(text='Δ (mm)', font=dict(color='#ddd')),
                          tickfont=dict(color='#ddd'), bgcolor='rgba(0,0,0,0)'),
            opacity=0.9),
        hovertext=[f"Δ {d:+.2f} mm" for d in color_mm], hoverinfo='text',
        name='Scan (Heatmap)',
    ))

    # Defekt-Cluster Marker
    if clusters:
        cx = [c['center'][0] * 1000 for c in clusters]
        cy = [c['center'][1] * 1000 for c in clusters]
        cz = [c['center'][2] * 1000 for c in clusters]
        labels = [f"<b>D{c['id']}</b><br>{c['type']}<br>max {c['max_dev_mm']:.2f} mm" for c in clusters]
        fig.add_trace(go.Scatter3d(
            x=cx, y=cy, z=cz, mode='markers+text',
            marker=dict(size=10, color='#fbbf24', symbol='diamond',
                        line=dict(color='#000', width=2)),
            text=[f"D{c['id']}" for c in clusters], textfont=dict(color='#fff', size=14),
            textposition='top center', hovertext=labels, hoverinfo='text',
            name='Defekte'))

    fig.update_layout(
        scene=dict(
            xaxis=dict(title='X (mm)', backgroundcolor='#0a0a0a', gridcolor='#222', color='#888'),
            yaxis=dict(title='Y (mm)', backgroundcolor='#0a0a0a', gridcolor='#222', color='#888'),
            zaxis=dict(title='Z (mm)', backgroundcolor='#0a0a0a', gridcolor='#222', color='#888'),
            aspectmode='data', bgcolor='#0a0a0a'),
        paper_bgcolor='#0a0a0a', font=dict(color='#bbb'),
        margin=dict(l=0, r=0, t=0, b=0), height=620,
        showlegend=True,
        legend=dict(bgcolor='rgba(0,0,0,0.5)', bordercolor='#333', borderwidth=1,
                    font=dict(color='#ddd'), x=0.02, y=0.98))
    plot_html = fig.to_html(include_plotlyjs='cdn', full_html=False, div_id='heatmap')

    # Histogramm
    hist = go.Figure(data=[go.Histogram(
        x=signed_dists * 1000, nbinsx=80,
        marker=dict(color='#3b82f6', line=dict(color='#1e3a8a', width=1)))])
    hist.add_vline(x=tolerance * 1000, line_dash='dash', line_color='#ef4444',
                   annotation_text=f"+{tolerance*1000:.1f}", annotation_font_color='#ef4444')
    hist.add_vline(x=-tolerance * 1000, line_dash='dash', line_color='#ef4444',
                   annotation_text=f"-{tolerance*1000:.1f}", annotation_font_color='#ef4444')
    hist.update_layout(
        xaxis=dict(title='Abweichung (mm)', gridcolor='#222', color='#888'),
        yaxis=dict(title='Punktanzahl', gridcolor='#222', color='#888'),
        paper_bgcolor='#0a0a0a', plot_bgcolor='#0a0a0a',
        font=dict(color='#bbb'), height=280, margin=dict(l=60, r=20, t=20, b=50))
    hist_html = hist.to_html(include_plotlyjs=False, full_html=False, div_id='hist')

    # Defekt-Tabelle
    if clusters:
        rows = ''.join([
            f'<tr><td><b>D{c["id"]}</b></td>'
            f'<td><span class="badge {"over" if c["type"]=="ueberschuessig" else "under"}">{c["type"]}</span></td>'
            f'<td>{c["n_points"]:,}</td>'
            f'<td><b>{c["max_dev_mm"]:.2f}</b> mm</td>'
            f'<td>{c["mean_dev_mm"]:.2f} mm</td>'
            f'<td>({c["center"][0]*1000:.1f}, {c["center"][1]*1000:.1f}, {c["center"][2]*1000:.1f})</td></tr>'
            for c in clusters])
    else:
        rows = ('<tr><td colspan="6" style="text-align:center; padding:30px; color:#10b981;">'
                'Keine Defekt-Cluster ueber der Toleranz</td></tr>')

    html = f"""<!DOCTYPE html>
<html lang="de"><head><meta charset="UTF-8"><title>CAD-Vergleich</title>
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ font-family: -apple-system, Segoe UI, sans-serif; background: #050505; color: #ddd; padding: 28px; }}
.container {{ max-width: 1400px; margin: 0 auto; }}
h1 {{ font-size: 30px; color: #fff; font-weight: 700; letter-spacing: -0.5px; }}
.meta {{ color: #777; font-size: 13px; margin: 6px 0 24px; }}
.banner {{ padding: 22px 28px; border-radius: 10px; background: {pass_color};
          color: #fff; font-size: 28px; font-weight: 800; margin-bottom: 24px;
          display: flex; justify-content: space-between; align-items: center;
          box-shadow: 0 4px 20px rgba(0,0,0,0.4); }}
.banner .sub {{ font-size: 15px; font-weight: 500; opacity: 0.9; }}
.grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; margin-bottom: 24px; }}
.stat {{ background: #0e0e0e; padding: 16px 18px; border-radius: 8px; border: 1px solid #1f1f1f; }}
.stat .label {{ color: #777; font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px; }}
.stat .value {{ font-size: 24px; font-weight: 600; color: #fff; margin-top: 6px; }}
.stat .sub {{ font-size: 12px; color: #666; }}
.panel {{ background: #0a0a0a; border: 1px solid #1f1f1f; border-radius: 10px;
         margin-bottom: 22px; overflow: hidden; }}
.panel h2 {{ padding: 16px 22px; font-size: 15px; border-bottom: 1px solid #1f1f1f;
            color: #fff; font-weight: 600; letter-spacing: 0.3px; }}
table {{ width: 100%; border-collapse: collapse; }}
th, td {{ padding: 12px 18px; text-align: left; border-bottom: 1px solid #161616; font-size: 14px; }}
th {{ background: #080808; color: #777; font-weight: 500; text-transform: uppercase;
     font-size: 11px; letter-spacing: 0.5px; }}
tr:hover {{ background: #0d0d0d; }}
.badge {{ padding: 4px 10px; border-radius: 4px; font-size: 11px; font-weight: 600;
         text-transform: uppercase; letter-spacing: 0.3px; }}
.badge.over {{ background: #7f1d1d; color: #fecaca; }}
.badge.under {{ background: #1e3a8a; color: #bfdbfe; }}
</style></head><body><div class="container">
  <h1>CAD-Vergleichsbericht</h1>
  <div class="meta">Scan: {os.path.basename(scan_path)} &nbsp;·&nbsp; CAD: {os.path.basename(cad_path)} &nbsp;·&nbsp; {datetime.now().strftime('%d.%m.%Y %H:%M')}</div>
  <div class="banner"><span>{pass_fail}</span><span class="sub">Toleranz ±{tolerance*1000:.2f} mm</span></div>
  <div class="grid">
    <div class="stat"><div class="label">Punkte gesamt</div><div class="value">{n_total:,}</div></div>
    <div class="stat"><div class="label">Defekt-Punkte</div><div class="value">{n_def:,}</div><div class="sub">{100*n_def/max(n_total,1):.1f} % der Wolke</div></div>
    <div class="stat"><div class="label">Defekt-Cluster</div><div class="value">{len(clusters)}</div></div>
    <div class="stat"><div class="label">Max. Abweichung</div><div class="value">{abs_d_mm.max():.2f} mm</div></div>
    <div class="stat"><div class="label">Mittel Abweichung</div><div class="value">{abs_d_mm.mean():.2f} mm</div></div>
    <div class="stat"><div class="label">Median Abweichung</div><div class="value">{np.median(abs_d_mm):.2f} mm</div></div>
    <div class="stat"><div class="label">RMSE Registrierung</div><div class="value">{rmse*1000:.2f} mm</div></div>
    <div class="stat"><div class="label">Fitness Registrierung</div><div class="value">{fitness:.3f}</div></div>
  </div>
  <div class="panel"><h2>3D-Heatmap (interaktiv – ziehen, zoomen, drehen)</h2>{plot_html}</div>
  <div class="panel"><h2>Verteilung der Abweichungen</h2>{hist_html}</div>
  <div class="panel"><h2>Defekt-Cluster</h2><table>
    <thead><tr><th>#</th><th>Typ</th><th>Punkte</th><th>Max. Δ</th><th>Mittel Δ</th><th>Position (mm)</th></tr></thead>
    <tbody>{rows}</tbody></table></div>
</div></body></html>"""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)


# --- HAUPTPROGRAMM ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='CAD-Vergleich Scan vs. STL')
    parser.add_argument('scan', help='Pfad zur Scan .ply')
    parser.add_argument('cad', help='Pfad zur CAD .stl')
    parser.add_argument('--tolerance-mm', type=float, default=DEFAULT_TOLERANCE_MM,
                        help=f'Toleranz in mm (default {DEFAULT_TOLERANCE_MM})')
    parser.add_argument('--cad-scale', type=float, default=None,
                        help='Manueller Skalierungsfaktor fuer das CAD '
                             '(0.001 = mm -> m, 0.01 = cm -> m). Default: auto')
    parser.add_argument('--output', help='Optionaler HTML-Output-Pfad')
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    tolerance = args.tolerance_mm / 1000.0

    if args.output:
        report_path = args.output
    else:
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = os.path.join(OUTPUT_DIR, f"report_{ts}.html")

    # 1. Laden
    step("1/9 Lade Daten")
    scan = o3d.io.read_point_cloud(args.scan)
    if len(scan.points) == 0:
        print(f"Fehler: Scan {args.scan} ist leer."); sys.exit(1)
    mesh = o3d.io.read_triangle_mesh(args.cad)
    if len(mesh.vertices) == 0:
        print(f"Fehler: CAD {args.cad} ist leer."); sys.exit(1)
    mesh.compute_vertex_normals()
    step(f"   Scan: {len(scan.points):,} Punkte | CAD: {len(mesh.triangles):,} Dreiecke")

    # 2. Scan saeubern (DBSCAN + SOR) - VOR Einheiten-Erkennung,
    #    damit Outlier nicht die Skalierungs-Logik kaputtmachen
    step("2/9 Scan saeubern (DBSCAN + SOR)")
    n_before = len(scan.points)
    scan = clean_scan(scan)
    scan.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.005, max_nn=30))
    step(f"   {n_before:,} -> {len(scan.points):,} Punkte")

    # 3. Einheiten
    step("3/9 Einheiten pruefen")
    mesh = detect_units_and_scale(mesh, scan, manual_scale=args.cad_scale)

    # 4. CAD samplen
    step(f"4/9 CAD samplen ({CAD_SAMPLE_POINTS:,} Punkte, Poisson Disk)")
    cad_pcd = sample_cad(mesh, CAD_SAMPLE_POINTS)

    # 5. Globale Registrierung (PCA + Multi-Hypothesen ICP)
    step("5/9 Globale Registrierung (PCA + Multi-Hypothesen ICP)")
    ransac = global_registration(cad_pcd, scan, VOXEL_COARSE)
    step(f"   Beste Pose: fitness {ransac.fitness:.3f} | RMSE {ransac.inlier_rmse*1000:.2f} mm")
    if ransac.fitness < 0.3:
        print("   WARNUNG: Globale Pose schlecht. Ergebnis vermutlich unbrauchbar.")
    T_global = ransac.transformation

    # 6. HPR
    step("6/9 Hidden Point Removal auf CAD")
    cad_in_scan = copy.deepcopy(cad_pcd).transform(T_global)
    cad_visible = hidden_point_removal(cad_in_scan, [0.0, 0.0, 0.0])
    step(f"   {len(cad_pcd.points):,} -> {len(cad_visible.points):,} sichtbare Punkte")

    # 7. Multi-Scale ICP
    step("7/9 Multi-Scale ICP")
    T_refine = np.eye(4)
    for vs, name in [(VOXEL_COARSE, '5 mm'), (VOXEL_MEDIUM, '1 mm'), (VOXEL_FINE, '0.3 mm')]:
        result = refine_icp(cad_visible, scan, T_refine, vs)
        T_refine = result.transformation
        step(f"   Voxel {name}: fitness {result.fitness:.3f} | RMSE {result.inlier_rmse*1000:.3f} mm")
    final_fitness = result.fitness
    final_rmse = result.inlier_rmse

    # CAD-Mesh in finales Scan-Frame
    T_total = T_refine @ T_global
    mesh_final = copy.deepcopy(mesh).transform(T_total)

    # 8. Signed distances
    step("8/9 Vorzeichenbehaftete Distanzen")
    scan_pts = np.asarray(scan.points)
    sd = signed_distances(scan_pts, mesh_final)
    step(f"   max |Δ| {np.abs(sd).max()*1000:.2f} mm | mean |Δ| {np.abs(sd).mean()*1000:.2f} mm")

    # 9. Defekt-Cluster
    step(f"9/9 Defekt-Cluster suchen (Toleranz {tolerance*1000:.2f} mm)")
    clusters, defect_pts, defect_d = find_defect_clusters(scan_pts, sd, tolerance)
    step(f"   {len(clusters)} Cluster gefunden")

    # HTML
    step("Erstelle HTML-Report")
    make_html_report(scan_pts, sd, clusters, tolerance, report_path,
                     args.scan, args.cad, final_fitness, final_rmse,
                     mesh_aligned=mesh_final)

    # Defekt-PLY
    if len(defect_pts) > 0:
        defect_pcd = o3d.geometry.PointCloud()
        defect_pcd.points = o3d.utility.Vector3dVector(defect_pts)
        colors = np.zeros((len(defect_pts), 3))
        colors[defect_d > 0] = [1.0, 0.2, 0.2]   # rot = ueberschuessig
        colors[defect_d < 0] = [0.2, 0.4, 1.0]   # blau = fehlend
        defect_pcd.colors = o3d.utility.Vector3dVector(colors)
        defect_path = report_path.replace('.html', '_defects.ply')
        o3d.io.write_point_cloud(defect_path, defect_pcd)
        step(f"Defekt-PLY: {defect_path}")

    pf = "PASS" if not clusters else "FAIL"
    print()
    print(f"================  ERGEBNIS: {pf}  ================")
    print(f"Report:  {report_path}")
    if clusters:
        worst = max(c['max_dev_mm'] for c in clusters)
        print(f"{len(clusters)} Defekt-Cluster, groesste Abweichung {worst:.2f} mm")
    print()


if __name__ == '__main__':
    main()
