import os
import cv2
import pandas as pd
from ultralytics import YOLO

# === Configuration ===
VIDEO_PATH = "data/videos/road_test3_2.mp4"
MODEL_PATH = "models/best.pt"
OUTPUT_VIDEO_PATH = "output/hazard_output_logged.mp4"
LOG_PATH = "output/detections.csv"
ALERT_CLASSES = {"debris", "pothole", "construction"}  # Add your custom class names
ALERT_DURATION = 15  # frames to keep alert visible

# === Ensure output directory exists ===
os.makedirs("output", exist_ok=True)

# === Load YOLOv8 model ===
print(f"📦 Loading YOLO model from {MODEL_PATH}")
model = YOLO(MODEL_PATH)

# === Open input video ===
print(f"🎥 Opening video file {VIDEO_PATH}")
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print("❌ Failed to open video.")
    exit()

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (width, height))

# === Initialize tracking variables ===
frame_index = 0
alert_active = False
alert_timer = 0
log_data = []

# === Detection loop with guaranteed cleanup ===
try:
    print("🚀 Starting detection...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("✅ End of video or read failure.")
            break

        print(f"🧠 Processing frame {frame_index}")
        results = model(frame)[0]
        boxes = results.boxes
        frame_alert = False

        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                cls_id = int(box.cls)
                conf = float(box.conf)
                label = model.names[cls_id]

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                color = (0, 255, 0)
                if label in ALERT_CLASSES:
                    color = (0, 0, 255)
                    frame_alert = True

                # Draw bounding box and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2
                )

                # Log detection
                log_data.append({
                    "frame": frame_index,
                    "label": label,
                    "confidence": round(conf, 3),
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2
                })

        # Manage alert overlay state
        if frame_alert:
            alert_active = True
            alert_timer = ALERT_DURATION
        elif alert_timer > 0:
            alert_timer -= 1
        else:
            alert_active = False

        # Add overlay if alert is active
        if alert_active:
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (width, height), (0, 0, 255), -1)
            frame = cv2.addWeighted(overlay, 0.25, frame, 0.75, 0)
            cv2.putText(
                frame, "!! HAZARD ALERT !!", (50, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 4
            )

        # Show live video
        cv2.imshow("Road Hazard Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("🛑 Interrupted by user.")
            break

        out.write(frame)
        frame_index += 1

finally:
    print(f"💾 Writing log to {LOG_PATH}")
    pd.DataFrame(log_data).to_csv(LOG_PATH, index=False)
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"✅ All done. Output saved to: {OUTPUT_VIDEO_PATH}")
