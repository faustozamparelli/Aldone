from ultralytics import YOLO
import cv2
import math

cap = cv2.VideoCapture(1)
cap.set(3, 640)
cap.set(4, 480)

model = YOLO("yolo-Weights/yolov8n.pt")

classNames = [
    "bottle",
    "wine glass",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
]


while True:
    success, img = cap.read()
    results = model(img, stream=True)

    # coordinates
    for r in results:
        boxes = r.boxes

        for box in boxes:
            # bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # convert to int values

            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            confidence = math.ceil((box.conf[0] * 100)) / 100
            print("Confidence --->", confidence)

            cls = int(box.cls[0])
            if cls in range(len(classNames)):
                print("Class name -->", classNames[cls])

            # object details
            org = [x1, y1]
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            color = (255, 0, 0)
            thickness = 2

            cv2.putText(
                img,
                classNames[cls] if cls in range(len(classNames)) else cls,
                org,
                font,
                fontScale,
                color,
                thickness,
            )

    cv2.imshow("Webcam", img)
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
