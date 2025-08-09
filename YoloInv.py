from ultralytics import YOLO
import torch
import cv2

# replace the path of the YOLO model that you want to test
#model = YOLO("/home/jetson/bucket/yolov5n_orange_object/weights/best.pt")
model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5n_orange_object/weights/best.engine'>

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 640) #480)

while True:
    SUCCESS, frame = cap.read()

    if not SUCCESS: break

    results = model(frame) # , conf=0.85)

    annotated_frame = results[0].plot() #.render()[0]

    cv2.imshow("YOLO Object Detection", annotated_frame)

    # Press 'q' to exit the video display
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
