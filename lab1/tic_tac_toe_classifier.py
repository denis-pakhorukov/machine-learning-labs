import csv

import numpy as np
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB

with open('tic_tac_toe.txt') as file:
    lines_count = sum(1 for line in file)

features_encoder = LabelEncoder()
features_encoder.fit(['o', 'x', 'b'])
labels_encoder = LabelEncoder()
labels_encoder.fit(['negative', 'positive'])

features_matrix = np.zeros((lines_count, 9), np.int8)
labels = np.zeros(lines_count, np.uint8)

with open('tic_tac_toe.txt') as file:
    for i, values in enumerate(csv.reader(file)):
        *features, label = values
        features_matrix[i] = features_encoder.transform(features)
        labels[i] = labels_encoder.transform([label])

test_case_count = 50
test_row_indexes = np.random.choice(features_matrix.shape[0], test_case_count, True)
test_feature_matrix = features_matrix[test_row_indexes, :]
test_labels = labels[test_row_indexes]

features_matrix = np.delete(features_matrix, test_row_indexes, 0)
labels = np.delete(labels, test_row_indexes)

model = GaussianNB()
model.fit(features_matrix, labels)

expected = test_labels
predicted = model.predict(test_feature_matrix)
print(metrics.accuracy_score(expected, predicted))
print(metrics.classification_report(expected, predicted, labels))
print(metrics.confusion_matrix(expected, predicted))
