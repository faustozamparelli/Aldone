from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

categories = ["talk.religion.misc", "sci.space", "comp.graphics"]

train = fetch_20newsgroups(subset="train", categories=categories)
test = fetch_20newsgroups(subset="test", categories=categories)

model = make_pipeline(TfidfVectorizer(), MultinomialNB())

model.fit(train.data, train.target)

labels = model.predict(test.data)

conf_matrix = confusion_matrix(test.target, labels)
sns.heatmap(
    conf_matrix.T,
    fmt="d",
    square=True,
    annot=True,
    cbar=False,
    xticklabels=train.target_names,
    yticklabels=train.target_names,
)
plt.xlabel("true labels")
plt.ylabel("predicted labels")
plt.show()

s = "I believe in the future"
pred = model.predict([s])
print(train.target_names[pred[0]])
