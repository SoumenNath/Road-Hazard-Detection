import cv2
from collections import deque
from ultralytics import YOLO

# Load your custom-trained model
model = YOLO("models/best.pt")

# Open the input video
cap = cv2.VideoCapture("data/videos/road_test3_2.mp4")

# Prepare to save the output video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(
    "output/hazard_output.mp4",
    fourcc,
    30.0,
    (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
)

# Buffer to hold recent detections (labels + boxes) to keep captions visible longer
label_buffer = deque(maxlen=10)  # Keep detections from last 10 frames

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    detections = results[0].boxes

    current_labels = []
    if detections is not None and len(detections) > 0:
        for box in detections:
            # Optional: filter low-confidence detections
            if box.conf[0] < 0.5:
                continue

            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            xyxy = box.xyxy[0].tolist()  # bounding box coordinates
            current_labels.append((label, xyxy))

    # Add current frame's labels to the buffer
    label_buffer.append(current_labels)

    # Draw all recent labels from the buffer on the frame
    for label_set in label_buffer:
        for label, xyxy in label_set:
            x1, y1, x2, y2 = map(int, xyxy)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2)  # orange box
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 165, 255), 2)

    # Show the frame with annotations
    cv2.imshow("Road Hazard Detection", frame)
    out.write(frame)

    # Press 'q' to quit early
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()
