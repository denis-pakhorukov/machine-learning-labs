import os
from collections import Counter

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB


def make_frequency_dict(dir):
    emails = [os.path.join(dir, filename) for filename in os.listdir(dir)]
    words = []

    for mail in emails:
        with open(mail) as mail_file:
            for line in mail_file:
                words += line.split()

    counter = Counter(words)

    keys = list(counter.keys())  # avoid dict change size during iteration
    for item in keys:
        if not item.isalpha() or len(item) < 2:
            del counter[item]

    return dict(counter.most_common(MOST_FREQUENT_WORDS_LIMIT))


def extract_features(dir):
    filenames = os.listdir(dir)
    features_matrix = np.zeros((len(filenames), len(dictionary)), np.uint32)
    labels = np.zeros(len(filenames), np.uint8)

    for file_index, filename in enumerate(filenames):
        labels[file_index] = filename.startswith('spmsg')
        path = os.path.join(dir, filename)
        with open(path) as file:
            for line_index, line in enumerate(file):
                if line_index != 2:
                    continue
                words_counter = Counter(line.split())
                for word_index, word in enumerate(dictionary):
                    if word in words_counter:
                        features_matrix[file_index, word_index] = words_counter[word]
    return features_matrix, labels


MOST_FREQUENT_WORDS_LIMIT = 3000
TRAIN_DIR = './emails-train-data'
TEST_DIR = './emails-test-data'

dictionary = make_frequency_dict(TRAIN_DIR)

features_matrix, labels = extract_features(TRAIN_DIR)
test_feature_matrix, test_labels = extract_features(TEST_DIR)

model = GaussianNB()
model.fit(features_matrix, labels)

predicted_labels = model.predict(test_feature_matrix)
print(accuracy_score(test_labels, predicted_labels))
