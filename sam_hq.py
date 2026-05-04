"""
Snapshot-Workflow: YOLO live + HQ-SAM beim Capture (Intel RealSense D405).

Setup:
  pip install segment-anything-hq
  sam_hq_vit_l.pth herunterladen von https://github.com/SysCV/sam-hq

Live: c = Capture, q = Beenden
Review: p = Pointcloud, s = Screenshot, r = zurueck zu Live, q = Beenden
"""

import os
import time

import cv2
import numpy as np
import open3d as o3d
import pyrealsense2 as rs
import torch
from segment_anything_hq import SamPredictor, sam_model_registry
from ultralytics import YOLO

# --- KONFIGURATION ---------------------------------------------------------
YOLO_MODEL_PATH = r"C:/PythonProject3/runs/segment/bauteil_seg-3/weights/best.pt"
HQSAM_MODEL_TYPE = "vit_h"                  # vit_tiny / vit_b / vit_l / vit_h
HQSAM_CHECKPOINT = "sam_hq_vit_h.pth"       # Pfad zum heruntergeladenen Checkpoint
CONF_THRESHOLD   = 0.8
EXPAND_PIXELS    = 3

OUTPUT_DIR = "pointclouds"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- MODELLE ---------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Nutze Gerät: {device}")

yolo_model = YOLO(YOLO_MODEL_PATH).to(device)
print("YOLO geladen.")

hqsam = sam_model_registry[HQSAM_MODEL_TYPE](checkpoint=HQSAM_CHECKPOINT)
hqsam.to(device=device)
hqsam_predictor = SamPredictor(hqsam)
print(f"HQ-SAM ({HQSAM_MODEL_TYPE}) geladen.")

# --- REALSENSE -------------------------------------------------------------
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
profile = pipeline.start(config)

depth_sensor = profile.get_device().first_depth_sensor()
depth_scale  = depth_sensor.get_depth_scale()
align = rs.align(rs.stream.color)


def expand_mask(mask: np.ndarray, pixels: int) -> np.ndarray:
    if pixels <= 0:
        return mask
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * pixels + 1, 2 * pixels + 1))
    return cv2.dilate(mask.astype(np.uint8), k, iterations=1).astype(bool)


def mask_to_pointcloud(mask, depth_image, color_image, intrinsics):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    z = depth_image[ys, xs].astype(np.float32) * depth_scale
    valid = z > 0
    xs, ys, z = xs[valid], ys[valid], z[valid]
    if len(xs) == 0:
        return None
    fx, fy = intrinsics.fx, intrinsics.fy
    cx, cy = intrinsics.ppx, intrinsics.ppy
    x = (xs - cx) * z / fx
    y = (ys - cy) * z / fy
    points = np.stack([x, y, z], axis=-1)
    colors = color_image[ys, xs][:, ::-1].astype(np.float32) / 255.0  # BGR -> RGB
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def run_inference(color_img_bgr):
    """YOLO -> HQ-SAM auf einem einzelnen Snapshot."""
    yolo_res = yolo_model.predict(color_img_bgr, conf=CONF_THRESHOLD,
                                  verbose=False, device=device)
    if len(yolo_res[0].boxes) == 0:
        return None, None, None, None

    boxes = yolo_res[0].boxes.xyxy.cpu().numpy()
    confs = yolo_res[0].boxes.conf.cpu().numpy()
    cls_ids = yolo_res[0].boxes.cls.cpu().numpy().astype(int)
    names_map = yolo_res[0].names
    cls_names = [names_map[i] for i in cls_ids]

    # HQ-SAM erwartet RGB
    img_rgb = cv2.cvtColor(color_img_bgr, cv2.COLOR_BGR2RGB)
    hqsam_predictor.set_image(img_rgb)

    combined = np.zeros(color_img_bgr.shape[:2], dtype=bool)
    for box in boxes:
        masks, scores, _ = hqsam_predictor.predict(
            box=np.array(box, dtype=np.float32),
            multimask_output=False,
            hq_token_only=False,   # True = nur HQ-Token, oft schaerfere Kanten
        )
        m = masks[0].astype(bool)
        m = expand_mask(m, EXPAND_PIXELS)
        combined |= m

    return combined, boxes, cls_names, confs


def render_overlay(color_img, mask, boxes, cls_names=None, confs=None):
    out = color_img.copy()
    if mask is not None and mask.any():
        overlay = np.zeros_like(out, dtype=np.uint8)
        overlay[mask] = [255, 0, 150]
        out = cv2.addWeighted(out, 1.0, overlay, 0.5, 0)
    if boxes is not None:
        for i, b in enumerate(boxes):
            x1, y1, x2, y2 = int(b[0]), int(b[1]), int(b[2]), int(b[3])
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
            if cls_names is not None and confs is not None:
                label = f"{cls_names[i]} {confs[i]:.2f}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                y_label = max(y1, th + 6)
                cv2.rectangle(out, (x1, y_label - th - 6),
                              (x1 + tw + 6, y_label), (0, 255, 0), -1)
                cv2.putText(out, label, (x1 + 3, y_label - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    return out


# --- MAIN LOOP -------------------------------------------------------------
print("Live: c = Capture, q = Beenden")
print("Review: p = Pointcloud, s = Screenshot, r = zurueck, q = Beenden")

mode = "live"
fps_counter, fps, last_time = 0, 0, time.time()
screenshot_count = 0
pcd_count = 0

snap_color = None
snap_depth = None
snap_intrinsics = None
snap_mask = None
snap_boxes = None
snap_cls_names = None
snap_confs = None
snap_overlay = None
snap_inference_ms = 0.0

try:
    while True:
        if mode == "live":
            frames = pipeline.wait_for_frames()
            aligned = align.process(frames)
            color = aligned.get_color_frame()
            depth = aligned.get_depth_frame()
            if not color or not depth:
                continue

            img = np.asanyarray(color.get_data())
            depth_image = np.asanyarray(depth.get_data())
            intrinsics = color.profile.as_video_stream_profile().intrinsics

            # Live nur YOLO-Boxen, kein SAM (waere zu langsam)
            live_res = yolo_model.predict(img, conf=CONF_THRESHOLD,
                                          verbose=False, device=device)
            live_boxes = live_cls_names = live_confs = None
            n_live = 0
            if len(live_res[0].boxes) > 0:
                live_boxes = live_res[0].boxes.xyxy.cpu().numpy()
                live_confs = live_res[0].boxes.conf.cpu().numpy()
                live_cls_ids = live_res[0].boxes.cls.cpu().numpy().astype(int)
                names_map = live_res[0].names
                live_cls_names = [names_map[i] for i in live_cls_ids]
                n_live = len(live_boxes)

            display_img = render_overlay(img, None, live_boxes,
                                         live_cls_names, live_confs)

            fps_counter += 1
            if time.time() - last_time >= 1.0:
                fps = fps_counter
                fps_counter = 0
                last_time = time.time()

            cv2.putText(display_img,
                        f"LIVE | FPS: {fps} | Boxen: {n_live} | c = Capture",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow("YOLO + HQ-SAM Snapshot", display_img)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                snap_color = img.copy()
                snap_depth = depth_image.copy()
                snap_intrinsics = intrinsics
                print("Capture - Inferenz laeuft (HQ-SAM)...")
                t0 = time.time()
                snap_mask, snap_boxes, snap_cls_names, snap_confs = run_inference(snap_color)
                snap_inference_ms = (time.time() - t0) * 1000
                snap_overlay = render_overlay(snap_color, snap_mask, snap_boxes,
                                              snap_cls_names, snap_confs)
                n = 0 if snap_mask is None else int(snap_mask.any())
                print(f"Inferenz fertig in {snap_inference_ms:.0f} ms - Maske: {n}")
                mode = "review"

        else:  # review
            view = snap_overlay.copy()
            n_mask = 0 if snap_mask is None else (1 if snap_mask.any() else 0)
            cv2.putText(view,
                        f"REVIEW HQ-SAM | {snap_inference_ms:.0f} ms | Maske: {n_mask} | "
                        f"p=Pointcloud  s=Screenshot  r=zurueck",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.imshow("YOLO + HQ-SAM Snapshot", view)

            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                mode = "live"
                last_time = time.time()
                fps_counter = 0
            elif key == ord('s'):
                cv2.imwrite(f"screenshot_{screenshot_count}.png", snap_overlay)
                print(f"Screenshot: screenshot_{screenshot_count}.png")
                screenshot_count += 1
            elif key == ord('p'):
                if snap_mask is None or not snap_mask.any():
                    print("Keine Maske vorhanden.")
                    continue
                pcd = mask_to_pointcloud(snap_mask, snap_depth,
                                         snap_color, snap_intrinsics)
                if pcd is None or len(pcd.points) == 0:
                    print("Punktwolke leer.")
                    continue
                out_path = os.path.join(OUTPUT_DIR, f"pointcloud_{pcd_count:03d}.ply")
                o3d.io.write_point_cloud(out_path, pcd)
                print(f"Gespeichert: {out_path}  ({len(pcd.points)} Punkte)")
                pcd_count += 1

finally:
    pipeline.stop()
    cv2.destroyAllWindows()