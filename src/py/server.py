from flask import Flask, request
from labler.nnlabler import is_it_a_question, TextClassifier

app = Flask(__name__)


@app.route("/")
def hello():
    return {"success": True, "abra": "cadabra"}


@app.route("/question_classifier", methods=["POST"])
def question_classifier():
    j = request.json
    text = j["text"]
    return {"is_it_a_question": is_it_a_question(text)}


if __name__ == "__main__":
    app.run(port=3001)
