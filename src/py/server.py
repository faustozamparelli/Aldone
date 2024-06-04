from flask import Flask, request
from question_classifier.nnlabler import TextClassifier, is_it_a_query
from todo_classifier.use_classifier import classify_todo
import json
import cv2
from flask import Response
from ultralytics import YOLO

app = Flask(__name__)


@app.route("/")
def hello():
    return {"success": True, "abra": "cadabra"}


@app.route("/question_classifier", methods=["POST"])
def question_classifier():
    j = request.json
    text = j["text"]
    return json.dumps({"is_it_a_query": is_it_a_query(text)})


@app.route("/todo_classifier", methods=["POST"])
def classify_todo_route():
    j = request.json
    text = j["text"]
    category = classify_todo(text)
    result = {"category": category}
    return json.dumps(result)


@app.route("/webcam")
def webcam_display():
    def webcam():

        model = YOLO("yolov8n.pt")

        camera = cv2.VideoCapture(0)
        camera.set(3, 640)
        camera.set(4, 480)

        foods = {
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
        }

        seen = set()

        while True:
            success, frame = camera.read()
            if not success:
                camera.release()

            results = model(frame, stream=True)
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    cls = r.names[int(box.cls[0])]
                    if cls not in foods:
                        break
                    if cls not in seen:
                        seen.add(cls)
                        with open(".seen", "w") as f:
                            f.write("\n".join(seen))
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)
                    cv2.putText(
                        frame,
                        cls,
                        [x1, y1],
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2,
                    )
                cv2.putText(
                    frame,
                    f"{seen}",
                    (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2,
                )

            ret, buffer = cv2.imencode(".jpg", frame)
            if not ret:
                continue
            frame = buffer.tobytes()
            yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")

    return Response(webcam(), mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    app.run(port=3001)
