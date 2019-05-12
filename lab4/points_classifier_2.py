import matplotlib.pyplot as plt
import mglearn
import numpy as np
from sklearn.svm import SVC

converters = {3: {b'green': 0, b'red': 1}.get}
samples = np.loadtxt('svmdata2.txt', delimiter='\t', skiprows=1,
                     usecols=(1, 2, 3), converters=converters)
X_train = samples[:, :-1]
y_train = np.array(samples[:, -1].transpose(), dtype=np.uint8)

samples = np.loadtxt('svmdata2test.txt', delimiter='\t', skiprows=1,
                     usecols=(1, 2, 3), converters=converters)
X_test = samples[:, :-1]
y_test = np.array(samples[:, -1].transpose(), dtype=np.uint8)

tests = (
    (483, '100% accuracy on Training set'),
    (1, '100% accuracy on Test set')
)
for C, title in tests:
    svc = SVC(kernel='linear', C=483)
    svc.fit(X_train, y_train)

    mglearn.plots.plot_2d_classification(svc, X_train, fill=True, alpha=.7)
    mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train)
    plt.legend(['Class 0', 'Class 1'], loc=(1.01, 0.3))
    plt.title(title)
    plt.show()

    print(title)
    print('Number of supporting vectors:', *svc.n_support_)
    print('Accuracy on training set:', svc.score(X_train, y_train))
    print('Accuracy on test set:', svc.score(X_test, y_test))
