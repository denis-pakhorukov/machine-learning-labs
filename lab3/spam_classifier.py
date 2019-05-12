import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier

label_encodings = {b'"n"': 0, b'"y"': 1}
samples = np.loadtxt('spam7.csv', delimiter=',',
                     skiprows=1, converters={6: label_encodings.get})
X = samples[:, :-1]
y = np.array(samples[:, -1].transpose(), dtype=np.uint8)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

grid_search = GridSearchCV(DecisionTreeClassifier(random_state=0), {
    'criterion': ('gini', 'entropy'),
    'max_depth': range(1, 20),
    'max_features': range(1, 7)
})
grid_search.fit(X_train, y_train)
print(grid_search.best_params_)
print(grid_search.best_score_)

tree = DecisionTreeClassifier(**grid_search.best_params_)
tree.fit(X_train, y_train)
print('Accuracy on training set: {:.3f}'.format(tree.score(X_train, y_train)))
print('Accuracy on test set: {:.3f}'.format(tree.score(X_test, y_test)))
