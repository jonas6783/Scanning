import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO, SAM
import torch

# --- KONFIGURATION ---
YOLO_MODEL_PATH = r"C:/PythonProject3/runs/segment/bauteil_seg-3/weights/best.pt"
SAM_MODEL_PATH = "sam2.1_t.pt"
CONF_THRESHOLD = 0.95

# ENTFERNUNGSFILTER (in Metern)
MIN_DIST = 0.05  # 5 cm
MAX_DIST = 0.18  # 18 cm - Alles dahinter wird schwarz!

# MASK_EXPANSION: Wie viele Pixel soll die Maske wachsen? (0 = original)
EXPAND_PIXELS = 2

# Hardware Setup
device = "cuda" if torch.cuda.is_available() else "cpu"
yolo_model = YOLO(YOLO_MODEL_PATH).to(device)
sam_model = SAM(SAM_MODEL_PATH).to(device)

# RealSense Setup
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
pipeline.start(config)

# Filter & Align
align = rs.align(rs.stream.color)
threshold_filter = rs.threshold_filter(min_dist=MIN_DIST, max_dist=MAX_DIST)

try:
    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()

        if not color_frame or not depth_frame:
            continue

        # 1. ENTFERNUNGSFILTER ANWENDEN
        depth_frame = threshold_filter.process(depth_frame)

        img = np.asanyarray(color_frame.get_data())
        depth_data = np.asanyarray(depth_frame.get_data())

        # Hintergrund im Bild schwarz machen für YOLO (Pre-Processing)
        img_prepped = img.copy()
        img_prepped[depth_data == 0] = 0

        # 2. YOLO FINDET BOXEN
        results = yolo_model.predict(img_prepped, conf=CONF_THRESHOLD, verbose=False, device=device)
        display_img = img.copy()

        if len(results[0].boxes) > 0:
            boxes = results[0].boxes.xyxy.cpu().numpy()

            # 3. SAM SEGMENTIERUNG
            sam_results = sam_model.predict(img, bboxes=boxes, verbose=False, device=device)

            for r in sam_results:
                if r.masks is not None:
                    for mask_data in r.masks.data:
                        mask = mask_data.cpu().numpy().astype(np.uint8)

                        # --- 4. MASKE MINIMAL VERGRÖSSERN (Dilation) ---
                        if EXPAND_PIXELS > 0:
                            kernel = np.ones((EXPAND_PIXELS, EXPAND_PIXELS), np.uint8)
                            mask = cv2.dilate(mask, kernel, iterations=1)

                        # Visualisierung
                        color_mask = np.zeros_like(display_img)
                        color_mask[mask > 0] = [0, 255, 0] # Grün
                        display_img = cv2.addWeighted(display_img, 1.0, color_mask, 0.4, 0)

        # UI
        cv2.imshow("D405: Distance Filter + YOLO + SAM + Dilation", display_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()