import cv2
import pandas as pd
from ultralytics import YOLO

# Config
ALERT_CLASSES = {"debris", "pothole", "construction"}
ALERT_DURATION = 15

def run_detection_and_log(input_video_path, output_video_path, csv_log_path):
    model = YOLO("models/best.pt")
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {input_video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    frame_idx = 0
    alert_active = False
    alert_timer = 0
    log_data = []

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame)[0]

            frame_alert = False
            if results.boxes is not None and len(results.boxes) > 0:
                for box in results.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    label = model.names[cls_id]

                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    color = (0, 255, 0)
                    if label in ALERT_CLASSES:
                        color = (0, 0, 255)
                        frame_alert = True

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(
                        frame,
                        f"{label} {conf:.2f}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        color,
                        2,
                    )

                    log_data.append(
                        {
                            "frame": frame_idx,
                            "label": label,
                            "confidence": round(conf, 3),
                            "x1": x1,
                            "y1": y1,
                            "x2": x2,
                            "y2": y2,
                        }
                    )

            if frame_alert:
                alert_active = True
                alert_timer = ALERT_DURATION
            elif alert_timer > 0:
                alert_timer -= 1
            else:
                alert_active = False

            if alert_active:
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (width, height), (0, 0, 255), -1)
                alpha = 0.25
                frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
                cv2.putText(
                    frame,
                    "!! HAZARD ALERT !!",
                    (50, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.5,
                    (255, 255, 255),
                    4,
                )

            try:
                cv2.imshow("Road Hazard Detection", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("[INFO] Quit key pressed. Exiting early.")
                    break
            except cv2.error as e:
                print(f"[WARN] OpenCV display error: {e}")
                break
            except ConnectionResetError as e:
                print(f"[WARN] Socket reset error: {e}")
                break

            out.write(frame)
            frame_idx += 1

    except KeyboardInterrupt:
        print("\n[INFO] Early termination detected. Saving progress...")

    finally:
        cap.release()
        out.release()
        try:
            cv2.destroyAllWindows()
        except:
            pass

        # Save logs
        if log_data:
            pd.DataFrame(log_data).to_csv(csv_log_path, index=False)
            print(f"[INFO] Detections saved to {csv_log_path}")
        else:
            pd.DataFrame(columns=["frame", "label", "confidence", "x1", "y1", "x2", "y2"]).to_csv(csv_log_path, index=False)
            print(f"[INFO] No detections found. Empty log saved to {csv_log_path}")
