import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

converters = {3: {b'green': 0, b'red': 1}.get}
samples = np.loadtxt('svmdata4.txt', delimiter='\t', skiprows=1,
                     usecols=(1, 2, 3), converters=converters)
X_train = samples[:, :-1]
y_train = np.array(samples[:, -1].transpose(), dtype=np.uint8)

samples = np.loadtxt('svmdata4test.txt', delimiter='\t', skiprows=1,
                     usecols=(1, 2, 3), converters=converters)
X_test = samples[:, :-1]
y_test = np.array(samples[:, -1].transpose(), dtype=np.uint8)

neighbors_settings = range(1, X_train.shape[0])
scores = []
for n_neighbors in neighbors_settings:
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    scores.append(knn.score(X_test, y_test))

print('optimal n_neighbors', neighbors_settings[scores.index(max(scores))])
plt.plot(neighbors_settings, scores)
plt.ylabel('Score')
plt.xlabel('n_neighbors')
plt.show()
