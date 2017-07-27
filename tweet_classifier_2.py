import sklearn.ensemble as sk
from sklearn import naive_bayes as nb
import pandas as pd
import numpy as np
import csv
df = pd.read_csv("./trump_tweets_ascii.csv")

trumps = []
for i in range(len(df)):
    trumps.append(0)

training_features = np.array(df)
training_labels = np.zeros(len(df))
print(training_features)

hash_dict = {}
train_set = []
for tweet in training_features:
    hashes = []
    for sentence in tweet:
        for word in sentence:
            hashofword = hash(word)
            if word not in hash_dict:
                hash_dict[word] = hashofword
                hashes.append(hashofword)
            else:
                hashes.append(hash_dict[word])
    train_set.append(hashes)
print(train_set)


#clf = sk.RandomForestRegressor(n_estimators=10, min_samples_split=3)
#clf.fit(train_set, training_labels)

clf = nb.GaussianNB()
clf.fit(train_set, training_labels)
