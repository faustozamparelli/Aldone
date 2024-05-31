import sys

import joblib
import torch
import torch.nn as nn


class TextClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(TextClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x


def classify_text(text):
    model = torch.load("src/py/question_classifier/label_model.pth")
    vectorizer = joblib.load("src/py/question_classifier/label_vectorizer.joblib")

    text_vec = vectorizer.transform([text]).toarray()
    text_tensor = torch.tensor(text_vec, dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        output = model(text_tensor)
        prediction_prob = output.squeeze().item()

    prediction = round(prediction_prob)

    return prediction, prediction_prob


def is_it_a_query(text):
    prediction, _ = classify_text(text)
    return prediction == 1


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py 'Your text here'")
        sys.exit(1)

    text = sys.argv[1]
    prediction, confidence = classify_text(text)
    print(f"Label: {prediction}")  # , Confidence: {confidence}"
