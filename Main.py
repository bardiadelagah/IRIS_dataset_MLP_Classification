from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from example import myClass
from sklearn.model_selection import validation_curve
from sklearn.model_selection import learning_curve
from sklearn.metrics import make_scorer, accuracy_score
import matplotlib.pyplot as plt
import numpy as np
from array import *
"""
This code make Find best parameters for IRIS dataset classification with MLP model
Show test and train dataset on plot
Show learning curve and Validataion curve
"""
# make an instance form example.py for using implication methods to answer questions
myInstance = myClass()

# load data set
iris = datasets.load_iris()
X = iris.data[:100, :2]
y = iris.target[:100]

# split data set to train and test part. %40 of data set use for test and %60 use for train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)

# show train an test data in two individual plots
myInstance.showDataSetOnPlot(X_train, y_train)
myInstance.showDataSetOnPlot(X_test, y_test)

# make a dict for GridSearchCV method
param_range = [5, 10, 20]
param_name = 'hidden_layer_sizes'
tuned_parameters = {param_name: param_range}

# make an MLP classifier for GridSearchCV method without given
mlp = MLPClassifier(activation='logistic',
                    solver='sgd',
                    learning_rate='adaptive',
                    learning_rate_init=0.1,
                    max_iter=1000,
                    random_state=1,
                    verbose=10,
                    )

# get model and parameters form GridSearchCV and
clf = GridSearchCV(estimator=mlp, param_grid=tuned_parameters, cv=5)

# train mlp classifier with [5, 10, 20] hidden layer neurons and 5-fold cross validation
clf.fit(X_train, y_train)

# get result
print("cv results")
print(clf.cv_results_)
print("best estimator")
print(clf.best_estimator_)
print("best parametrs")
print(clf.best_params_)
print("number of output layer")
print(clf.best_estimator_.n_outputs_)
means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

# show loss curve for best estimator
myInstance.showLossCurvePlot(clf.best_estimator_)


mymeans = means.tolist()
# get loss for all items in param_range.
myloss = myInstance.getAllLoss(mlp=mlp, param_range=param_range, param_name=param_name, cv=5, X_train=X_train,
                               y_train=y_train)
# show
myInstance.showLossAndMeansOnPlot(param_range, myloss, mymeans)


train_errors = clf.score(X_train, y_train)
print("train_errors")
print(train_errors)
test_error = clf.score(X_test, y_test)
print("test_error")
print(test_error)

# Generating a meshgrid
x_lable_min = X[:, 0].min()
x_lable_max = X[:, 0].max()
y_lable_min = X[:, 1].min()
y_lable_max = X[:, 1].max()
xx, yy = np.meshgrid(np.arange(x_lable_min, x_lable_max, 0.1), np.arange(y_lable_min, y_lable_max, 0.1))
ax = plt.subplot(1, 1, 1)
Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
Z = Z.reshape(xx.shape)
plt.scatter(X[0:49, 0], X[0:49, 1], marker='*', color='g')
plt.scatter(X[50:100, 0], X[50:100, 1], marker='^', color='r')
ax.contour(xx, yy, Z, alpha=.8)
plt.show()
