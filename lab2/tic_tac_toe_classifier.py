import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

feature_encodings = {b'o': 0, b'x': 1, b'b': 2}
label_encodings = {b'negative': 0, b'positive': 1}
converters = dict.fromkeys(range(0, 9), feature_encodings.get)
converters[9] = label_encodings.get

samples = np.loadtxt('tic_tac_toe.txt', delimiter=',', dtype=np.uint8, converters=converters)
X = samples[:, :-1]
y = samples[:, -1].transpose()

test_sizes = np.concatenate((np.arange(0.01, 0.9, 0.05),
                             np.arange(0.9, 1, 0.01)))
scores = []
for test_size in test_sizes:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=1549294)

    knn = KNeighborsClassifier()
    knn.fit(X_train, y_train)

    scores.append(knn.score(X_test, y_test))
    print('{}/{} ({}) score {:.2f}'.format(
        y_train.shape[0], y_test.shape[0], test_size, scores[-1]))

plt.plot(test_sizes, scores)
plt.show()
