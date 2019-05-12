from io import StringIO

import graphviz
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz

samples = np.loadtxt('lenses.txt', usecols=range(1, 6), dtype=np.uint8)
X = samples[:, :-1]
y = np.array(samples[:, -1].transpose(), dtype=np.uint8)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

tree = DecisionTreeClassifier(random_state=0)
tree.fit(X_train, y_train)
print('Decision Tree with default settings')
print('Accuracy on training set: {:.3f}'.format(tree.score(X_train, y_train)))
print('Accuracy on test set: {:.3f}'.format(tree.score(X_test, y_test)))

sample = np.array([[2, 1, 2, 1]])
print('A patient who has presbyopia, myopia, astigmatism, shortened tears '
      'should wear lenses of type', tree.predict(sample)[0])

tree_io = StringIO()
export_graphviz(tree, out_file=tree_io, class_names='hard soft none'.split(),
                feature_names='presbyopia, myopia, astigmatism, shortened tears'.split(', '),
                impurity=False, filled=True)
graphviz.Source(tree_io.getvalue()).view()
tree_io.close()
