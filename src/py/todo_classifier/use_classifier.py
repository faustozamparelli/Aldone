import pickle
from train_classifier import target_names

with open("src/py/todo_classifier/model.pkl", "rb") as f:
    model = pickle.load(f)


def classify_todo(text):
    return model.predict([text])[0]


if __name__ == "__main__":
    while True:
        s = input("Enter a sentence: ")
        pred = model.predict([s])
        print(target_names[pred[0]])
