from matplotlib import pyplot as plt
import numpy as np

from clearml import Task, Logger

task = Task.init(project_name='SF_M2.1_04_1_HyperParams', task_name='HyperParams')

parameters = {
    'n_train' : 150,
    'n_test' : 1000,
    'noise' : 0.1,
    'max_depth' :5,
    'random_state' : 17
}

parameters = task.connect(parameters)

def f(x):
    x = x.ravel()
    return np.exp(-x ** 2) + 1.5 * np.exp(-(x - 2) ** 2)

def generate(n_samples, noise):
    X = np.random.rand(n_samples) * 10 - 5
    X = np.sort(X).ravel()
    y = np.exp(-X ** 2) + 1.5 * np.exp(-(X - 2) ** 2) + \
        np.random.normal(0.0, noise, n_samples)
    X = X.reshape((n_samples, 1))
    return X, y

X_train, y_train = generate(n_samples=parameters['n_train'], noise=parameters['noise'])
X_test, y_test = generate(n_samples=parameters['n_test'], noise=parameters['noise'])

from sklearn.tree import DecisionTreeRegressor

reg_tree = DecisionTreeRegressor(max_depth=parameters['max_depth'], random_state=parameters['random_state'])

reg_tree.fit(X_train, y_train)
reg_tree_pred = reg_tree.predict(X_test)

plt.figure(figsize=(10, 6))
plt.plot(X_test, f(X_test), 'b')
plt.scatter(X_train, y_train, c='b', s=20)
plt.plot(X_test, reg_tree_pred, 'g', lw=2)
plt.xlim([-5, 5])
plt.title("Decision tree regressor, MSE = %.2f" % np.sum((y_test - reg_tree_pred) ** 2))
plt.show()