"""
Multi-View Registration: mehrere Single-View PLYs zu einer 3D-Wolke verschmelzen.

Voraussetzung:
- Captures aus verschiedenen Winkeln des gleichen Bauteils
- Bauteil zwischen Captures um 30-60 Grad drehen
- Captures in der Reihenfolge der Drehung speichern (pointcloud_000.ply, _001.ply, ...)
- Idealerweise einmal rundum drehen (letzte Aufnahme nahe der ersten -> Loop Closure)

Nutzung:
  python merge_views.py pointclouds/pointcloud_000.ply pointclouds/pointcloud_001.ply ...
  python merge_views.py "pointclouds/*.ply"
  python merge_views.py "pointclouds/*.ply" --output merged.ply --voxel 0.001
"""

import argparse
import glob
import os
import sys
from datetime import datetime

import numpy as np
import open3d as o3d

# --- KONFIGURATION ---------------------------------------------------------
VOXEL_SIZE             = 0.001    # 1 mm Voxel-Groesse fuer Registrierung
MAX_CORR_COARSE        = 0.015    # 15 mm fuer ICP-Grobstufe
MAX_CORR_FINE          = 0.003    # 3 mm fuer ICP-Feinstufe
LOOP_CLOSURE_MIN_FIT   = 0.30     # Loop-Closure-Kanten nur wenn fitness >= dieser Wert
SOR_NEIGHBORS          = 20
SOR_STD_RATIO          = 2.0


def step(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def expand_paths(args):
    """Glob-Patterns expandieren (Windows PowerShell macht das nicht automatisch)."""
    paths = []
    for s in args:
        if any(c in s for c in '*?['):
            paths.extend(sorted(glob.glob(s)))
        else:
            paths.append(s)
    return paths


def load_and_clean(path):
    """Laden + Outlier-Removal + Normalen."""
    pcd = o3d.io.read_point_cloud(path)
    if len(pcd.points) == 0:
        return None
    pcd, _ = pcd.remove_statistical_outlier(
        nb_neighbors=SOR_NEIGHBORS, std_ratio=SOR_STD_RATIO)
    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=VOXEL_SIZE * 5, max_nn=30))
    return pcd


def pairwise_registration(source, target, voxel):
    """Coarse-to-fine ICP mit Identity-Init - funktioniert wenn Wolken nahe beieinander sind."""
    icp_coarse = o3d.pipelines.registration.registration_icp(
        source, target, MAX_CORR_COARSE, np.identity(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50))
    icp_fine = o3d.pipelines.registration.registration_icp(
        source, target, MAX_CORR_FINE, icp_coarse.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50))
    info = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
        source, target, MAX_CORR_FINE, icp_fine.transformation)
    return icp_fine.transformation, info, icp_fine.fitness, icp_fine.inlier_rmse


def build_pose_graph(pcds_down):
    """Pose Graph: sequenzielle Kanten (immer) + Loop-Closure-Kanten (nur wenn ICP konvergiert)."""
    pose_graph = o3d.pipelines.registration.PoseGraph()
    odometry = np.identity(4)
    pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(odometry))

    n = len(pcds_down)
    n_loop = 0
    for source_id in range(n):
        for target_id in range(source_id + 1, n):
            T, info, fit, rmse = pairwise_registration(
                pcds_down[source_id], pcds_down[target_id], VOXEL_SIZE)

            if target_id == source_id + 1:
                # Sequenzielle Kante - immer hinzufuegen
                step(f"   seq  {source_id}->{target_id}: "
                     f"fitness {fit:.3f} | RMSE {rmse*1000:.2f} mm")
                odometry = np.dot(T, odometry)
                pose_graph.nodes.append(
                    o3d.pipelines.registration.PoseGraphNode(np.linalg.inv(odometry)))
                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(
                        source_id, target_id, T, info, uncertain=False))
            elif fit >= LOOP_CLOSURE_MIN_FIT:
                # Loop closure - nur wenn ICP plausibel
                step(f"   loop {source_id}->{target_id}: "
                     f"fitness {fit:.3f} | RMSE {rmse*1000:.2f} mm")
                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(
                        source_id, target_id, T, info, uncertain=True))
                n_loop += 1
            # else: zu wenig Ueberlapp, Kante weglassen
    return pose_graph, n_loop


def main():
    parser = argparse.ArgumentParser(
        description='Mehrere Single-View PLYs zu einer 3D-Wolke verschmelzen.')
    parser.add_argument('scans', nargs='+',
                        help='Pfade zu Scan-PLYs (mind. 2). Glob-Patterns wie "*.ply" erlaubt.')
    parser.add_argument('--output', default='merged.ply',
                        help='Output-PLY (default: merged.ply)')
    parser.add_argument('--voxel', type=float, default=VOXEL_SIZE,
                        help=f'Voxel-Groesse fuer Down-Sampling in m (default {VOXEL_SIZE})')
    args = parser.parse_args()

    scan_paths = expand_paths(args.scans)
    if len(scan_paths) < 2:
        print(f"Mindestens 2 Scans noetig. Gefunden: {len(scan_paths)}")
        sys.exit(1)

    step(f"Lade {len(scan_paths)} Scans:")
    for p in scan_paths:
        print(f"   {p}")

    pcds = []
    for p in scan_paths:
        pcd = load_and_clean(p)
        if pcd is None or len(pcd.points) < 100:
            print(f"   WARNUNG: {p} leer/zu klein, ueberspringe.")
            continue
        pcds.append(pcd)
        print(f"   {os.path.basename(p)}: {len(pcd.points):,} Punkte")

    if len(pcds) < 2:
        print("Zu wenig brauchbare Scans.")
        sys.exit(1)

    step(f"Down-Sampling auf {args.voxel*1000:.1f} mm")
    pcds_down = [p.voxel_down_sample(args.voxel) for p in pcds]
    for p in pcds_down:
        p.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=args.voxel * 5, max_nn=30))

    step("Paarweise Registrierung (ICP)")
    pose_graph, n_loop = build_pose_graph(pcds_down)
    step(f"   {n_loop} Loop-Closure-Kanten gefunden")

    step("Globale Pose-Graph Optimierung (Levenberg-Marquardt)")
    option = o3d.pipelines.registration.GlobalOptimizationOption(
        max_correspondence_distance=MAX_CORR_FINE,
        edge_prune_threshold=0.25,
        reference_node=0)
    o3d.pipelines.registration.global_optimization(
        pose_graph,
        o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
        o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
        option)

    step("Wolken transformieren und mergen")
    merged = o3d.geometry.PointCloud()
    for i, pcd in enumerate(pcds):
        pcd_t = o3d.geometry.PointCloud(pcd)
        pcd_t.transform(pose_graph.nodes[i].pose)
        merged += pcd_t

    # Final voxel-downsample, damit die Punktedichte gleichmaessig wird
    n_before = len(merged.points)
    merged_clean = merged.voxel_down_sample(args.voxel)
    step(f"Merged: {n_before:,} -> {len(merged_clean.points):,} Punkte (nach Final-Downsample)")

    # Final SOR auf das Gesamtergebnis
    merged_clean, _ = merged_clean.remove_statistical_outlier(
        nb_neighbors=SOR_NEIGHBORS, std_ratio=SOR_STD_RATIO)
    merged_clean.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=args.voxel * 5, max_nn=30))

    step(f"Speichere {args.output}")
    o3d.io.write_point_cloud(args.output, merged_clean)

    print()
    print(f"Fertig: {len(merged_clean.points):,} Punkte in {args.output}")
    print(f"Naechster Schritt:")
    print(f"  python cad_compare.py {args.output} bauteil.stl")


if __name__ == '__main__':
    main()
