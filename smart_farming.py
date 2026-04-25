import os
import glob
from ultralytics import YOLO

# 1. Initialize the YOLO model
print("--- Initializing AI Agriculture Model ---")
model = YOLO('yolov8n.pt') 

def run_smart_system(dataset_path, moisture):
    print("\n" + "="*55)
    print("  AI-POWERED PEST DETECTION & INTELLIGENT IRRIGATION")
    print("="*55)

    # --- MODULE 1: AI PEST DETECTION ---
    # This finds all .jpg images inside your new subfolders (healthy, curl_virus, etc.)
    image_list = glob.glob(os.path.join(dataset_path, "**/*.jpg"), recursive=True)
    
    if image_list:
        print(f"[AI Vision] Dataset Found! {len(image_list)} images detected.")
        # Process the first 3 images for your demonstration
        for i in range(min(3, len(image_list))):
            img = image_list[i]
            print(f"  > Analyzing: {os.path.basename(img)}")
            # Results are saved in 'runs/detect/predict'
            model.predict(source=img, save=True, conf=0.25)
        print(f"\n[SUCCESS] Visual results saved in 'runs/detect/predict'")
    else:
        print(f"[ERROR] No images found at: {dataset_path}")
        print("Ensure the path below exactly matches your 'pest_data' folder.")

    # --- MODULE 2: INTELLIGENT IRRIGATION ---
    THRESHOLD = 30.0
    print(f"\n[IoT Sensor] Current Soil Moisture: {moisture}%")
    if moisture < THRESHOLD:
        print(">> STATUS: LOW MOISTURE. ACTIVATING SMART IRRIGATION PUMP.")
    else:
        print(">> STATUS: MOISTURE OPTIMAL. PUMP REMAINING OFF.")
    print("="*55 + "\n")

if __name__ == "__main__":
    # --- THIS PATH MUST MATCH YOUR SCREENSHOT ---
    MY_FOLDER = r'C:\Users\LikhilCGowda\OneDrive\Desktop\SmartFarmProject\pest_data' 
    
    run_smart_system(dataset_path=MY_FOLDER, moisture=22.5)