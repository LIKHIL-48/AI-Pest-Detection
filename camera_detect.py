import cv2
import numpy as np
import json
from tensorflow.keras.models import load_model

# Load model
model = load_model("plant_model.h5")

# Load labels
with open("labels.json", "r") as f:
    class_labels = json.load(f)

class_labels = {v: k for k, v in class_labels.items()}

# Start camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Resize image
    img = cv2.resize(frame, (128, 128))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)
    confidence = np.max(prediction) * 100
    class_index = np.argmax(prediction)

    # -------- SCREEN DETECTION --------
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_ratio = np.sum(edges > 0) / (frame.shape[0] * frame.shape[1])

    # -------- GREEN CHECK --------
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_green = np.array([25, 40, 40])
    upper_green = np.array([90, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    green_ratio = cv2.countNonZero(mask) / (frame.shape[0] * frame.shape[1])

    # -------- FINAL DECISION --------
    if edge_ratio > 0.12:
        text = "Screen detected - Invalid input"

    elif green_ratio < 0.15:
        text = "Not a plant"

    elif confidence < 90:
        text = "Low confidence - Try again"

    else:
        disease = class_labels[class_index]
        disease = disease.replace("___", " - ")\
                         .replace("__", " ")\
                         .replace("_", " ")

        if "healthy" in disease.lower():
            disease = "Healthy leaf"

        text = f"{disease} ({confidence:.2f}%)"

    # Display
    cv2.putText(frame, text, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Plant Disease Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()