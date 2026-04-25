import os
from ultralytics import YOLO

# Load the AI model (First run will download the yolov8n.pt file)
print("--- Loading AI Model ---")
model = YOLO('yolov8n.pt') 

def run_system(img_file, moisture):
    print("\n" + "="*40)
    print("  SMART FARMING SYSTEM: LIVE REPORT")
    print("="*40)

    # MODULE 1: AI PEST DETECTION
    if os.path.exists(img_file):
        print(f"[AI Vision] Analyzing {img_file}...")
        # Runs detection and saves the output to a 'runs' folder
        results = model.predict(source=img_file, save=True, conf=0.25)
        print(f"[AI Vision] Done! Check 'runs/detect/predict' for results.")
    else:
        print(f"[AI Vision] Error: {img_file} not found in this folder.")

    # MODULE 2: INTELLIGENT IRRIGATION
    # Logic based on your 'Integrated AI-Driven Automation' report
    THRESHOLD = 30.0
    print(f"\n[IoT Sensor] Soil Moisture: {moisture}%")
    if moisture < THRESHOLD:
        print(">> STATUS: LOW MOISTURE. ACTIVATING PUMP.")
    else:
        print(">> STATUS: MOISTURE OPTIMAL. PUMP OFF.")
    print("="*40)

if __name__ == "__main__":
    # Test with a simulated moisture level
    run_system('test.jpg', moisture=24.5)