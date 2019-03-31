import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

label_encodings = {b'"nonspam"': 0, b'"spam"': 1}

X = np.loadtxt('spam.csv', delimiter=',', skiprows=1, usecols=range(1, 58))
y = np.loadtxt('spam.csv', delimiter=',', skiprows=1, usecols=58,
               dtype=np.uint8, converters={58: label_encodings.get})

test_sizes = np.concatenate((np.arange(0.01, 0.9, 0.05),
                             np.arange(0.9, 1, 0.01)))
scores = []
for test_size in test_sizes:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    knn = KNeighborsClassifier()
    knn.fit(X_train, y_train)

    scores.append(knn.score(X_test, y_test))
    print('{}/{} ({}) score {:.2f}'.format(
        y_train.shape[0], y_test.shape[0], test_size, scores[-1]))

plt.plot(test_sizes, scores)
plt.show()
