from textblob.classifiers import NaiveBayesClassifier
from sklearn import naive_bayes as nb
import pandas as pd
import numpy as np
import nltk
import hashlib
from nltk.tokenize import sent_tokenize, word_tokenize
df = pd.read_csv("./tweets.csv")

clf = nb.GaussianNB(priors=None)
df_train = df.drop(df.index[1])
df_test = df.drop(df.index[0])

training_features = np.array(df_train.drop(["1", "30"], 1))
training_labels = np.array(df_train["30"])
test_features = np.array(df_test.drop(["1", "30"], 1))
test_labels = np.array(df_test["30"])

training_features_hashed = []
for i in training_features:
    for j in i:
        try:
            if (float(j) == 0.0):
                training_features_hashed.append(1.0)
            else:
                training_features_hashed.append(float(j))
        except ValueError as e:
            training_features_hashed.append(1.0)

test_features_hashed = []
for i in test_features:
    for j in i:
        try:
            if (float(j) == 0.0):
                training_features_hashed.append(1.0)
            else:
                test_features_hashed.append(float(j))
        except ValueError as e:
            test_features_hashed.append(1.0)

print(training_features_hashed)
print(test_features_hashed)

#clf.fit(training_features_hashed, training_labels)
#print(clf.predict(test_features_hashed))

obama_1 = "Hi everybody! Back to the original handle. Is this thing still on? Michelle and I are off on a quick vacation, then we’ll get back to work."
obama_2 = "Obamacare has helped millions of Americans gain the peace of mind that comes with coverage. Show your support:"
obama_3 = "This is what happens when we focus on building an economy that works for everyone, not just those at the top."
trump_1 = "victory and cannot be burdened with the tremendous medical costs and disruption that transgender in the military would entail. Thank you"
trump_2 = "Will be traveling to the Great State of Ohio tonight. Big crowd expected. See you there!"
trump_3 = "Sleazy Adam Schiff, the totally biased Congressman looking into Russia, spends all of his time on television pushing the Dem loss excuse!"
'''
train = [(obama_1, "obama"), (obama_2, "obama"), (trump_1, "trump"), (trump_2, "trump")]
test = [(obama_3, "obama"), (trump_3, "trump")]
'''
