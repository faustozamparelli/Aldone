from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import pickle

dataset = load_dataset("clinc_oos", "plus")
train, test, validation = dataset["train"], dataset["test"], dataset["validation"]

model = make_pipeline(TfidfVectorizer(), MultinomialNB())

train_data, train_target = train["text"], train["intent"]
model.fit(train_data, train_target)

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

test_data, test_target = test["text"], test["intent"]
labels = model.predict(test_data)

print(f"Accuracy: {100 * sum(test_target == labels) / len(test_target):.2f}%")

# TODO: conf matrix only shopping_list, shopping_list update, todo_list, todo_list update vs everything else altogether

# conf_matrix = confusion_matrix(test_target, labels)
# sns.heatmap(
#     conf_matrix.T,
#     fmt="d",
#     square=True,
#     annot=True,
#     cbar=False,
#     xticklabels=target_names,
#     yticklabels=target_names,
# )
# plt.xlabel("true labels")
# plt.ylabel("predicted labels")
# plt.show()
