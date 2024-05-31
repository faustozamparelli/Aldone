from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from use_classifier import target_names

dataset = load_dataset("clinc_oos", "imbalanced")
train, test, validation = dataset["train"], dataset["test"], dataset["validation"]

model = make_pipeline(TfidfVectorizer(), MultinomialNB())

train_data, train_target = train["text"], train["intent"]
model.fit(train_data, train_target)

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

test_data, test_target = test["text"], test["intent"]
labels = model.predict(test_data)

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
