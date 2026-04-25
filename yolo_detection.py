from ultralytics import YOLO
import cv2

# Load YOLO model (will auto download)
model = YOLO("yolov8n.pt")

# Load your image
img = cv2.imread(r"C:\Users\LikhilCGowda\OneDrive\Desktop\PROJECT FOLDER\plant.jpg")

# Run detection
results = model(img)

# Show result
results[0].show()