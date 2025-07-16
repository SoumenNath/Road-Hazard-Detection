import cv2
import os

cap = cv2.VideoCapture("data/videos/road_test1.mp4")
output_folder = "data/images/train"
os.makedirs(output_folder, exist_ok=True)

frame_num = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    if frame_num % 50 == 0:  # Save every 50th frame
        cv2.imwrite(f"{output_folder}/frame_{frame_num}.jpg", frame)
    frame_num += 1

cap.release()