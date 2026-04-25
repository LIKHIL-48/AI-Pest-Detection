import cv2
import numpy as np

img = cv2.imread("plant.jpg")
img = cv2.resize(img, (800, 600))

# Convert to HSV (better for color detection)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Define range for damaged (brown/yellow spots)
lower = np.array([10, 50, 50])
upper = np.array([30, 255, 255])

mask = cv2.inRange(hsv, lower, upper)

# Remove noise
kernel = np.ones((5,5), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

# Find contours
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

count = 0

for cnt in contours:
    area = cv2.contourArea(cnt)

    # Better filtering
    if 1000 < area < 8000:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,0,255), 2)
        count += 1

# ALERT SYSTEM
if count > 0:
    print("⚠️ PEST DETECTED!")

cv2.imshow("Detection", img)
cv2.imshow("Mask", mask)

cv2.waitKey(0)
cv2.destroyAllWindows()