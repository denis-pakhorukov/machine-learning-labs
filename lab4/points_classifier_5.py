import matplotlib.pyplot as plt
import mglearn
import numpy as np
from sklearn.svm import SVC

converters = {3: {b'green': 0, b'red': 1}.get}
samples = np.loadtxt('svmdata5.txt', delimiter='\t', skiprows=1,
                     usecols=(1, 2, 3), converters=converters)
X_train = samples[:, :-1]
y_train = np.array(samples[:, -1].transpose(), dtype=np.uint8)

samples = np.loadtxt('svmdata5test.txt', delimiter='\t', skiprows=1,
                     usecols=(1, 2, 3), converters=converters)
X_test = samples[:, :-1]
y_test = np.array(samples[:, -1].transpose(), dtype=np.uint8)

for kernel in ('poly', 'rbf', 'sigmoid'):
    svc = SVC(kernel=kernel)
    svc.fit(X_train, y_train)

    mglearn.plots.plot_2d_classification(svc, X_train, fill=True, alpha=.7)
    mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train)
    plt.legend(['Class 0', 'Class 1'], loc=(1.01, 0.3))
    plt.title(kernel)
    plt.show()

    print(kernel)
    print('Accuracy on training set:', svc.score(X_train, y_train))
    print('Accuracy on test set:', svc.score(X_test, y_test))

fig, axes = plt.subplots(3, 3, figsize=(15, 10))

for ax, gamma in zip(axes, [[1, 5, 9], [10, 50, 90], [100, 500, 1000]]):
    for a, gamma in zip(ax, gamma):
        svc = SVC(kernel='rbf', gamma=gamma, random_state=0)
        svc.fit(X_train, y_train)

        mglearn.plots.plot_2d_classification(svc, X_train, fill=True, alpha=.7, ax=a)
        mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train, ax=a)
        a.set_title('gamma = {}'.format(gamma))

        print('gamma =', gamma)
        print('Accuracy on training set:', svc.score(X_train, y_train))
        print('Accuracy on test set:', svc.score(X_test, y_test))

fig.show()
