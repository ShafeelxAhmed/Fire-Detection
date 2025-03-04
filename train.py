from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO("yolov8n.pt")  # Using the nano model (you can use yolov8s/m/l/x)

# Train the model on your dataset
model.train(
    data="C:/Users/shafeel/smoke detection/FireandSmokeDetection-2/data.yaml",  # Path to the data.yaml file
    epochs=70,  # Number of training epochs
    imgsz=640,  # Image size
    batch=8,  # Adjust batch size as per your GPU capability
    device="cuda"  # Use GPU for training
)
