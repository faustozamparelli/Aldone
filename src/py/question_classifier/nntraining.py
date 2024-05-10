import json

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

with open("./nndata.json", "r") as f:
    data = json.load(f)

texts = []
labels = []  # 1 for question, 0 for statement

for item in data:
    texts.append(item["text"])
    labels.append(item["label"])

labels = np.array(labels)

X, X_test, y, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42, stratify=labels
)

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train).toarray()
X_val_vec = vectorizer.transform(X_val).toarray()
X_test_vec = vectorizer.transform(X_test).toarray()

X_train_tensor = torch.tensor(X_train_vec, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val_vec, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_vec, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)


class TextDataset(Dataset):
    def __init__(self, x_tensor, y_tensor):
        self.x = x_tensor
        self.y = y_tensor

    def __getitem__(self, index):
        return (self.x[index], self.y[index])

    def __len__(self):
        return len(self.y)


train_dataset = TextDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

val_dataset = TextDataset(X_val_tensor, y_val_tensor)
val_loader = DataLoader(val_dataset, batch_size=32)

test_dataset = TextDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=32)


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


input_size = X_train_tensor.shape[1]
hidden_size = 64
output_size = 1
model = TextClassifier(input_size, hidden_size, output_size)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    loss = None
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(1), labels)
        loss.backward()
        optimizer.step()

    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            val_loss = criterion(outputs.squeeze(), labels)
            total_val_loss += val_loss.item()

    if loss is not None:
        print(
            f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {loss.item():.4f}, Validation Loss: {total_val_loss/len(val_loader):.4f}"
        )

model.eval()
correct = 0
total = 0
threshold = 0.5
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        predicted_probs = outputs.squeeze()
        predicted_labels = (predicted_probs > threshold).float()
        total += labels.size(0)
        correct += (predicted_labels == labels).sum().item()

print(f"Accuracy on test set: {100 * correct / total:.2f}%")

torch.save(model, "./label_model.pth")
joblib.dump(vectorizer, "./label_vectorizer.joblib")
