import csv

import numpy as np
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB

with open('spam.csv') as file:
    features_count = -1
    reader = csv.reader(file)
    for row in reader:
        if features_count == -1:
            features_count = len(row) - 2  # do not count first and last columns
    lines_count = reader.line_num - 1

features_matrix = np.zeros((lines_count, features_count), np.float32)
labels = np.zeros(lines_count, np.int8)

with open('spam.csv') as file:
    for i, row in enumerate(csv.reader(file), start=-1):
        if i == -1:
            continue
        index, *features, label = row
        features_matrix[i] = features
        labels[i] = 1 if label.strip() == 'spam' else -1

test_case_count = 100
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
