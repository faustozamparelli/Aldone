# import the dataset and the modules
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# get a subset of categories
categories = ["talk.religion.misc", "sci.space", "comp.graphics"]

# load the dataset
train = fetch_20newsgroups(subset="train", categories=categories)
test = fetch_20newsgroups(subset="test", categories=categories)

# prepare the data and create the model
model = make_pipeline(TfidfVectorizer(), MultinomialNB())

# train the model
model.fit(train.data, train.target)

# test the model
labels = model.predict(test.data)

# plot the confusion matrix
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

# try the classifier with our text
s = "I believe in the future"
pred = model.predict([s])
print(train.target_names[pred[0]])
