from io import StringIO

import graphviz
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz

converters = {10: lambda val: val[1:-1]}  # remove quotes from class label
samples = np.loadtxt('glass.csv', delimiter=',', skiprows=1,
                     usecols=range(1, 11), converters=converters)
X = samples[:, :-1]
y = np.array(samples[:, -1].transpose(), dtype=np.uint8)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

tree = DecisionTreeClassifier(random_state=0)
tree.fit(X_train, y_train)
print('Decision Tree with default settings')
print('Accuracy on training set: {:.3f}'.format(tree.score(X_train, y_train)))
print('Accuracy on test set: {:.3f}'.format(tree.score(X_test, y_test)))

tree_io = StringIO()
export_graphviz(tree, out_file=tree_io, class_names='1 2 3 4 5 6 7'.split(),
                feature_names='Ri Na Mg Al Si K Ca Ba Fe'.split(), impurity=False, filled=True)
graphviz.Source(tree_io.getvalue()).view()
tree_io.close()

tests = [
    ('criterion', ('gini', 'entropy')),
    ('max_depth', range(1, 13)),
    ('max_features', range(1, 10)),
    ('min_samples_split', range(2, 50)),
]
for param, settings in tests:
    training_accuracy = []
    test_accuracy = []
    for i in settings:
        kwargs = {param: i, 'random_state': 0}
        tree = DecisionTreeClassifier(**kwargs)
        tree.fit(X_train, y_train)
        training_accuracy.append(tree.score(X_train, y_train))
        test_accuracy.append(tree.score(X_test, y_test))

    plt.plot(settings, training_accuracy, label='training accuracy')
    plt.plot(settings, test_accuracy, label='test accuracy')
    plt.show()
