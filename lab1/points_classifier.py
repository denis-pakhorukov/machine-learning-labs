import matplotlib.pyplot as plt
import numpy as np
from numpy.random import normal
from sklearn.naive_bayes import GaussianNB


def generate_data_set(mean1, mean2, stdev1, mean3, mean4, stdev2, size):
    a = np.array(list(zip(normal(mean1, stdev1, size), normal(mean2, stdev1, size))), np.float32)
    b = np.array(list(zip(normal(mean3, stdev2, size), normal(mean4, stdev2, size))), np.float32)
    return np.concatenate((a, b)), np.array([-1] * size + [1] * size, np.int8)


args = dict(mean1=10, mean2=14, stdev1=4, mean3=20, mean4=18, stdev2=3)
X_train, y_train = generate_data_set(**args, size=50)
X_test, y_test = generate_data_set(**args, size=50)

bayes = GaussianNB()
bayes.fit(X_train, y_train)
print(bayes.score(X_test, y_test))

plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.Set1)
plt.show()
