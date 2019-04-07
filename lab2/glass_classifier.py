import csv

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

converters = {10: lambda val: val[1:-1]}  # remove quotes from class label
samples = np.loadtxt('glass.csv', delimiter=',', skiprows=1,
                     usecols=range(1, 11), converters=converters)
X = samples[:, :-1]
y = np.array(samples[:, -1].transpose(), dtype=np.uint8)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=15424)

training_accuracy = []
test_accuracy = []
neighbors_settings = range(1, 101)
for n_neighbors in neighbors_settings:
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    training_accuracy.append(knn.score(X_train, y_train))
    test_accuracy.append(knn.score(X_test, y_test))

plt.plot(neighbors_settings, training_accuracy, label='training accuracy')
plt.plot(neighbors_settings, test_accuracy, label='test accuracy')
plt.ylabel('Accuracy')
plt.xlabel('n_neighbors')
plt.legend()
plt.show()

metrics = (('euclidean', None), ('manhattan', None), ('chebyshev', None), ('minkowski', {'p': 1}))
for metric, params in metrics:
    knn = KNeighborsClassifier(metric=metric, metric_params=params)
    knn.fit(X_train, y_train)
    print(metric, 'training accuracy', knn.score(X_train, y_train))
    print(metric, 'test accuracy', knn.score(X_test, y_test))

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
sample = np.array([[1.516, 11.7, 1.01, 1.19, 72.59, 0.43, 11.44, 0.02, 0.1]])
[prediction] = knn.predict(sample)
print(prediction)

with open('glass.csv') as f:
    chemical_elements = next(csv.reader(f))[1:-1]

scores = []
for i, element in enumerate(chemical_elements):
    reduced_X_train = np.delete(X_train, i, 1)
    reduced_X_test = np.delete(X_test, i, 1)
    knn = KNeighborsClassifier()
    knn.fit(reduced_X_train, y_train)
    scores.append(knn.score(reduced_X_test, y_test))

fig, ax = plt.subplots()
ax.barh(range(len(chemical_elements)), scores)
ax.set_yticklabels(chemical_elements)
ax.invert_yaxis()
ax.set_xlabel('Score')
plt.show()

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
