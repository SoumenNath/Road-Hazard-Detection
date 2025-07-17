from ultralytics import YOLO

# Load a base YOLOv8 model (choose size: n, s, m, l)
model = YOLO("models/yolov8n.pt")  # 'n' = nano, fastest model

# Train the model on your dataset
model.train(
    data="data/data.yaml",    # path to your dataset config
    epochs=15,                # how many times your model will see your entire training dataset during training - adjust as needed
    imgsz=640,                # image size for training - means 640x640 pixels
    batch=4,                  # how many images the model processes at once before updating the weights - Use smaller batch size on CPU
    name="road_hazard_train", # experiment name
    project="runs",           # output directory
    verbose=True,
    patience=5,               # early stopping if no improvement
    save=True,                # save checkpoints
    cache=True,               # cache images for faster training
    device="cpu"              # use GPU (0), or "cpu" if you don't have a CUDA compatable one

)
