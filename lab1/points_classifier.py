import matplotlib.pyplot as plt
import numpy as np
from numpy.random import normal
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB


def generate_data_set(mean1, mean2, stdev1, mean3, mean4, stdev2, size):
    a = np.array(list(zip(normal(mean1, stdev1, size), normal(mean2, stdev1, size))), np.float32)
    b = np.array(list(zip(normal(mean3, stdev2, size), normal(mean4, stdev2, size))), np.float32)
    return np.concatenate((a, b)), np.array([-1] * size + [1] * size, np.int8)


args = dict(mean1=10, mean2=14, stdev1=4, mean3=20, mean4=18, stdev2=3)
size = 50
test_size = 100
feature_matrix, labels = generate_data_set(**args, size=size)
test_feature_matrix, test_labels = generate_data_set(**args, size=test_size)

model = GaussianNB()
model.fit(feature_matrix, labels)
print(metrics.accuracy_score(test_labels, model.predict(test_feature_matrix)))

fig, ax = plt.subplots()
plt.plot(feature_matrix[:size], feature_matrix[size:], 'o')
plt.show()
