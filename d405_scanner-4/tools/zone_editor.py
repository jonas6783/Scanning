"""
tools/zone_editor.py
Interaktiver Zonen-Editor – zeigt CAD-STL + Toleranz-Zonen in Open3D.
Zonen werden als farbige Drahtgitter-Boxen dargestellt.
Linien verbinden die Zonen-Mittelpunkte zur Orientierung.

Verwendung:
    python tools/zone_editor.py --scan mein_bauteil.stl

Steuerung im Fenster:
    Maus    → Drehen / Zoomen
    Q       → Beenden und Zonen als Python-Code ausgeben
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

try:
    import open3d as o3d
except ImportError:
    print("open3d nicht installiert: pip install open3d")
    sys.exit(1)


# ──────────────────────────────────────────────────────────────────────
# Zonen-Definition (aus deinen gepickten Punkten abgeleitet)
# Koordinaten in mm, relativ zum Bauteil-Ursprung
# Passe x_min/max, y_min/max, z_min/max nach Bedarf an
# ──────────────────────────────────────────────────────────────────────
ZONES = [
    {
        "name":     "Flansch oben-rechts (neg Z)",
        "color":    [1.0, 0.2, 0.2],
        "tol_ok":   0.3,
        "tol_warn": 0.6,
        "x": (27.0, 35.0),
        "y": (-36.0, -7.0),
        "z": (-11.0, -6.5),
    },
    {
        "name":     "Flansch oben-rechts (pos Z)",
        "color":    [1.0, 0.4, 0.1],
        "tol_ok":   0.3,
        "tol_warn": 0.6,
        "x": (27.0, 35.0),
        "y": (-36.0, -7.0),
        "z": (7.5, 11.0),
    },
    {
        "name":     "Flansch oben-links (pos Z)",
        "color":    [1.0, 0.6, 0.0],
        "tol_ok":   0.3,
        "tol_warn": 0.6,
        "x": (27.0, 35.0),
        "y": (7.0, 36.0),
        "z": (7.0, 11.0),
    },
    {
        "name":     "Flansch oben-links (neg Z)",
        "color":    [0.9, 0.8, 0.0],
        "tol_ok":   0.3,
        "tol_warn": 0.6,
        "x": (27.0, 35.0),
        "y": (7.0, 36.0),
        "z": (-11.0, -6.5),
    },
    {
        "name":     "Mitte oben (Funktionsfläche)",
        "color":    [0.0, 0.9, 0.2],
        "tol_ok":   0.2,
        "tol_warn": 0.5,
        "x": (25.0, 35.0),
        "y": (-7.5, 7.5),
        "z": (-10.0, 8.0),
    },
    {
        "name":     "Schrägfläche unten (neg Z)",
        "color":    [0.0, 0.7, 0.9],
        "tol_ok":   0.5,
        "tol_warn": 1.0,
        "x": (2.0, 27.0),
        "y": (-7.0, 7.0),
        "z": (-17.0, -7.5),
    },
    {
        "name":     "Schrägfläche unten (pos Z)",
        "color":    [0.2, 0.5, 1.0],
        "tol_ok":   0.5,
        "tol_warn": 1.0,
        "x": (-1.0, 27.0),
        "y": (-7.0, 7.0),
        "z": (7.0, 17.0),
    },
    {
        "name":     "Flanke links (pos Z)",
        "color":    [0.6, 0.0, 1.0],
        "tol_ok":   0.8,
        "tol_warn": 1.5,
        "x": (-5.0, 35.0),
        "y": (24.0, 37.0),
        "z": (6.0, 8.0),
    },
    {
        "name":     "Flanke links (neg Z)",
        "color":    [0.7, 0.2, 0.9],
        "tol_ok":   0.8,
        "tol_warn": 1.5,
        "x": (-5.0, 34.0),
        "y": (23.0, 37.0),
        "z": (-13.0, -7.0),
    },
    {
        "name":     "Flanke rechts (neg Z)",
        "color":    [0.5, 0.1, 0.8],
        "tol_ok":   0.8,
        "tol_warn": 1.5,
        "x": (-5.0, 35.0),
        "y": (-37.0, -24.0),
        "z": (-11.0, -6.5),
    },
    {
        "name":     "Flanke rechts (pos Z)",
        "color":    [0.4, 0.0, 0.7],
        "tol_ok":   0.8,
        "tol_warn": 1.5,
        "x": (-5.0, 35.0),
        "y": (-37.0, -26.0),
        "z": (7.5, 10.5),
    },
    {
        "name":     "Hintere Kante links",
        "color":    [0.0, 0.8, 0.6],
        "tol_ok":   0.8,
        "tol_warn": 1.5,
        "x": (-18.0, -15.0),
        "y": (16.0, 21.0),
        "z": (-12.0, 13.0),
    },
    {
        "name":     "Hintere Kante rechts",
        "color":    [0.0, 0.6, 0.5],
        "tol_ok":   0.8,
        "tol_warn": 1.5,
        "x": (-23.0, -17.0),
        "y": (-20.0, -16.0),
        "z": (-12.0, 13.0),
    },
    {
        "name":     "Flanke unten rechts",
        "color":    [0.9, 0.5, 0.5],
        "tol_ok":   1.0,
        "tol_warn": 2.0,
        "x": (-17.0, -4.0),
        "y": (-26.0, -20.0),
        "z": (-8.0, 7.0),
    },
    {
        "name":     "Flanke unten links",
        "color":    [0.8, 0.5, 0.8],
        "tol_ok":   1.0,
        "tol_warn": 2.0,
        "x": (-13.0, -4.0),
        "y": (22.0, 24.0),
        "z": (-6.0, 7.0),
    },
]


# ──────────────────────────────────────────────────────────────────────
# Geometrie-Hilfsmethoden
# ──────────────────────────────────────────────────────────────────────

def make_zone_box(zone: dict, scale: float = 1.0) -> o3d.geometry.LineSet:
    """
    Erstellt einen Drahtgitter-Quader für eine Zone.
    scale: 1.0 wenn Koordinaten schon in der richtigen Einheit,
           0.001 wenn mm -> Meter umgerechnet werden soll.
    """
    x0, x1 = zone["x"][0] * scale, zone["x"][1] * scale
    y0, y1 = zone["y"][0] * scale, zone["y"][1] * scale
    z0, z1 = zone["z"][0] * scale, zone["z"][1] * scale

    pts = np.array([
        [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
        [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
    ])
    lines = [
        [0,1],[1,2],[2,3],[3,0],
        [4,5],[5,6],[6,7],[7,4],
        [0,4],[1,5],[2,6],[3,7],
    ]
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(pts)
    ls.lines  = o3d.utility.Vector2iVector(lines)
    ls.colors = o3d.utility.Vector3dVector([zone["color"]] * len(lines))
    return ls


def make_center_lines(zones: list, scale: float = 1.0) -> o3d.geometry.LineSet:
    """Verbindet Zonen-Mittelpunkte mit grauer Linie."""
    centers = []
    for z in zones:
        cx = (z["x"][0] + z["x"][1]) / 2 * scale
        cy = (z["y"][0] + z["y"][1]) / 2 * scale
        cz = (z["z"][0] + z["z"][1]) / 2 * scale
        centers.append([cx, cy, cz])

    centers = np.array(centers)
    lines   = [[i, i+1] for i in range(len(centers)-1)]

    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(centers)
    ls.lines  = o3d.utility.Vector2iVector(lines)
    ls.colors = o3d.utility.Vector3dVector([[0.4, 0.4, 0.4]] * len(lines))
    return ls


def make_center_spheres(zones: list, scale: float = 1.0, radius: float = 0.5) -> list:
    """Kleine Kugeln an Zonen-Mittelpunkten."""
    spheres = []
    for z in zones:
        cx = (z["x"][0] + z["x"][1]) / 2 * scale
        cy = (z["y"][0] + z["y"][1]) / 2 * scale
        cz = (z["z"][0] + z["z"][1]) / 2 * scale
        s = o3d.geometry.TriangleMesh.create_sphere(radius=radius * scale)
        s.translate([cx, cy, cz])
        s.paint_uniform_color(z["color"])
        s.compute_vertex_normals()
        spheres.append(s)
    return spheres


# ──────────────────────────────────────────────────────────────────────
# Code-Ausgabe
# ──────────────────────────────────────────────────────────────────────

def print_settings_code(zones: list, scale_to_meters: bool = True):
    """Gibt fertigen Python-Code für config/settings.py aus."""
    factor = 0.001 if scale_to_meters else 1.0
    print("\n" + "=" * 65)
    print("ZONEN-CODE FÜR config/settings.py  (Koordinaten in Metern)")
    print("=" * 65)
    print("zones: List[ToleranceZone] = field(default_factory=lambda: [")
    for z in zones:
        print(f"    ToleranceZone(")
        print(f"        name=\"{z['name']}\",")
        print(f"        tolerance_ok_m={z['tol_ok']*factor:.4f},    # {z['tol_ok']:.1f} mm")
        print(f"        tolerance_warn_m={z['tol_warn']*factor:.4f},  # {z['tol_warn']:.1f} mm")
        print(f"        x_min={z['x'][0]*factor:.4f}, x_max={z['x'][1]*factor:.4f},")
        print(f"        y_min={z['y'][0]*factor:.4f}, y_max={z['y'][1]*factor:.4f},")
        print(f"        z_min={z['z'][0]*factor:.4f}, z_max={z['z'][1]*factor:.4f},")
        print(f"    ),")
    print("])")
    print("=" * 65)


# ──────────────────────────────────────────────────────────────────────
# Hauptfunktion
# ──────────────────────────────────────────────────────────────────────

def run_editor(file_path: Path):
    print(f"Lade Datei: {file_path}")

    suffix = file_path.suffix.lower()
    scale  = 1.0   # Einheit der Datei (mm oder m)

    if suffix in (".stl", ".obj"):
        mesh = o3d.io.read_triangle_mesh(str(file_path))
        if len(mesh.vertices) == 0:
            print(f"FEHLER: Datei leer oder nicht lesbar: {file_path}")
            sys.exit(1)
        mesh.compute_vertex_normals()
        mesh.paint_uniform_color([0.75, 0.75, 0.80])

        # Einheit erkennen: wenn BBox > 500 → wahrscheinlich mm
        verts  = np.asarray(mesh.vertices)
        extent = verts.max(axis=0) - verts.min(axis=0)
        bbox_max = float(np.max(extent))

        if bbox_max > 0.5:
            # STL in mm – Zonen auch in mm → scale=1.0, kein Umrechnen
            scale = 1.0
            print(f"STL in mm erkannt (BBox max={bbox_max:.1f} mm)")
        else:
            # STL in Metern – Zonen in mm → scale=0.001
            scale = 0.001
            print(f"STL in Metern erkannt (BBox max={bbox_max*1000:.1f} mm)")

        pcd = mesh.sample_points_uniformly(number_of_points=300_000)
        pcd.paint_uniform_color([0.75, 0.75, 0.80])
        geometries = [mesh]
        print(f"STL geladen: {len(mesh.vertices):,} Vertices")

    else:
        pcd = o3d.io.read_point_cloud(str(file_path))
        if len(pcd.points) == 0:
            print(f"FEHLER: Datei leer: {file_path}")
            sys.exit(1)

        pts    = np.asarray(pcd.points)
        extent = pts.max(axis=0) - pts.min(axis=0)
        bbox_max = float(np.max(extent))
        scale = 1.0 if bbox_max > 0.5 else 0.001

        if not pcd.has_colors():
            pcd.paint_uniform_color([0.75, 0.75, 0.80])
        geometries = [pcd]
        print(f"PLY geladen: {len(pcd.points):,} Punkte, BBox max={bbox_max:.3f}")

    # Zonen aufbauen
    sphere_radius = 0.8 if scale == 1.0 else 0.0008

    for zone in ZONES:
        geometries.append(make_zone_box(zone, scale=scale))

    geometries.append(make_center_lines(ZONES, scale=scale))
    geometries.extend(make_center_spheres(ZONES, scale=scale, radius=sphere_radius))

    # Koordinatenachsen
    axis_size = 10.0 if scale == 1.0 else 0.01
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=axis_size)
    geometries.append(axes)

    # Legende in Konsole
    print("\nLegende (Zonen):")
    for i, z in enumerate(ZONES):
        c = z["color"]
        try:
            bar = f"\033[38;2;{int(c[0]*255)};{int(c[1]*255)};{int(c[2]*255)}m■\033[0m"
        except Exception:
            bar = "■"
        print(f"  {bar} {i+1:2d}. {z['name']:40s}  OK={z['tol_ok']:.1f}mm  WARN={z['tol_warn']:.1f}mm")

    print("\nSteuerung:")
    print("  Maus-Links  → Drehen")
    print("  Scroll      → Zoomen")
    print("  Maus-Rechts → Verschieben")
    print("  Q           → Beenden + Code ausgeben\n")

    # Visualizer
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="D405 Zonen-Editor", width=1500, height=900)

    for g in geometries:
        vis.add_geometry(g)

    opt = vis.get_render_option()
    opt.background_color  = np.array([0.07, 0.07, 0.10])
    opt.point_size        = 1.5
    opt.line_width        = 2.5
    opt.mesh_show_back_face = True

    def quit_callback(vis):
        vis.destroy_window()
        return False

    vis.register_key_callback(ord("Q"), quit_callback)
    vis.register_key_callback(ord("q"), quit_callback)

    vis.run()
    vis.destroy_window()

    # Code ausgeben
    scale_to_m = (scale == 1.0)  # wenn mm → in Meter umrechnen für settings.py
    print_settings_code(ZONES, scale_to_meters=scale_to_m)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Zonen-Editor – Toleranz-Zonen am CAD-Bauteil visualisieren und verfeinern"
    )
    parser.add_argument(
        "--scan", type=Path, required=True,
        help="Pfad zur STL/PLY-Datei (z.B. mein_bauteil.stl)"
    )
    args = parser.parse_args()

    if not args.scan.exists():
        print(f"FEHLER: Datei nicht gefunden: {args.scan}")
        sys.exit(1)

    run_editor(args.scan)
