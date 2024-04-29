import sys

import joblib
import torch
import torch.nn as nn


# Define your TextClassifier class here
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


# Load the model and the vectorizer
model = torch.load("./label_model.pth")
vectorizer = joblib.load("./label_vectorizer.joblib")


def predict_query(query, threshold=0.7):
    # Vectorize the query
    query_vec = vectorizer.transform([query]).toarray()

    # Convert to PyTorch tensor
    query_tensor = torch.tensor(query_vec, dtype=torch.float32)

    # Make prediction
    model.eval()
    with torch.no_grad():
        output = model(query_tensor)
        prediction_prob = output.squeeze().item()

    # Apply threshold
    prediction = 1 if prediction_prob > threshold else 0

    return prediction, prediction_prob


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py 'Your question here'")
        sys.exit(1)

    query = sys.argv[1]
    prediction, confidence = predict_query(query)
    print(f"Prediction: {prediction}, Confidence: {confidence}")
