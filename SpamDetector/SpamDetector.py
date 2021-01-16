# %%
# Imports
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import recall_score, precision_score, f1_score

# %%
# constants
JSON_PATH = 'SpamData/01_Processing/email-text-data.json'

# %%
# Retreiving the Data
data = pd.read_json(JSON_PATH)
data.sort_index(inplace=True)
# %%
# Retrieving the features and vocabulary
vectorizer = CountVectorizer(stop_words='english')
all_features = vectorizer.fit_transform(data.MESSAGE)
print(all_features.shape)
print(vectorizer.vocabulary_)
xtrain, xtest, ytrain, ytest = train_test_split(
    all_features, data.CATEGORY, test_size=0.3, random_state=88)
print(xtrain.shape)

# %%
# Training the classifier
classifier = MultinomialNB()
classifier.fit(xtrain, ytrain)
correct_eval = (ytest == classifier.predict(xtest)).sum()
print(correct_eval)
wrong_eval = (ytest.size-correct_eval)
print(wrong_eval)
accuracy = classifier.score(xtest, ytest)*100
# Metric Calculations
print(round(accuracy, 2))
recall_score(ytest, classifier.predict(xtest))
precision_score(ytest, classifier.predict(xtest))
f1_score(ytest, classifier.predict(xtest))
