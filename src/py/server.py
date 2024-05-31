from flask import Flask, request
from question_classifier.nnlabler import TextClassifier, is_it_a_query
from todo_classifier.use_classifier import classify_todo
import json

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
def extract_todo():
    j = request.json
    text = j["text"]
    category = classify_todo(text)
    result = {"category": category}
    return json.dumps(result)


if __name__ == "__main__":
    app.run(port=3001)
