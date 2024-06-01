from flask import Flask, request
from question_classifier.nnlabler import TextClassifier, is_it_a_query
from todo_classifier.use_classifier import classify_todo
import json
import cv2
from flask import Response

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
def classify_todo():
    j = request.json
    text = j["text"]
    category = classify_todo(text)
    result = {"category": category}
    return json.dumps(result)


@app.route("/webcam")
def webcam_display():
    def webcam():
        camera = cv2.VideoCapture(2)

        while True:
            success, frame = camera.read()
            if success:

                ret, buffer = cv2.imencode(".jpg", frame)
                frame = buffer.tobytes()
                yield (
                    b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
                )
            else:
                camera.release()

    return Response(webcam(), mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    app.run(port=3001)
