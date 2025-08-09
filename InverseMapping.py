import cv2
import numpy as np

import mediapipe
# === CONFIGURATION ===
KNOWN_DISTANCE = 50.0  # cm – measure the red object placed at this distance from the camera
KNOWN_WIDTH = 10.0     # cm – real width of the red object

# === Function to detect the red object and return its width in pixels ===
def find_red_object_width(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    

    # Red color range in HSV
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([179, 255, 255])
    
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, mask
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # Draw bounding box
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    return w, mask

# === Focal Length Calculation ===
def calculate_focal_length(known_distance, known_width, width_in_image):
    return (width_in_image * known_distance) / known_width

# === Distance Estimation ===
def estimate_distance(focal_length, known_width, width_in_image):
    return (known_width * focal_length) / width_in_image

# === MAIN PROGRAM ===
cap = cv2.VideoCapture(0)

print("📷 Starting camera...")
print("🧪 Place the red object at EXACTLY", KNOWN_DISTANCE, "cm to calibrate focal length.")
print("Press 'c' to calibrate, then 'q' to quit anytime.\n")

focal_length = None

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to read from webcam.")
        break

    width_in_pixels, mask = find_red_object_width(frame)

    if width_in_pixels:
        if focal_length:
            distance = estimate_distance(focal_length, KNOWN_WIDTH, width_in_pixels)
            cv2.putText(frame, f"Distance: {distance:.2f} cm", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        else:
            cv2.putText(frame, "Press 'c' to calibrate", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Show both frames
    cv2.imshow("Distance Estimator", frame)
    cv2.imshow("Red Mask", mask)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c') and width_in_pixels:
        focal_length = calculate_focal_length(KNOWN_DISTANCE, KNOWN_WIDTH, width_in_pixels)
        print(f"✅ Calibrated focal length: {focal_length:.2f} px")

cap.release()
cv2.destroyAllWindows()
