import cv2
import numpy as np

# Load thermal image
img = cv2.imread('1.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Threshold to detect hot regions
_, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

# Stronger noise reduction to merge body parts
kernel = np.ones((15,15), np.uint8)
binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
binary = cv2.dilate(binary, kernel, iterations=2)

# Find blobs
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Filter for person-sized blobs only
hotspot_count = 0
for contour in contours:
    area = cv2.contourArea(contour)
    if area > 2000:  # Larger minimum area
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = h / w if w > 0 else 0
        
        # Filter by aspect ratio (person shape)
        if 0.5 < aspect_ratio < 3:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            hotspot_count += 1
            cv2.putText(img, str(hotspot_count), (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

cv2.putText(img, f'Total Hotspots: {hotspot_count}', (10, 40),
           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

cv2.imshow('Binary Mask', binary)
cv2.imshow('Detected Hotspots', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

print(f"Detected {hotspot_count} people")