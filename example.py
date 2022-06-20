from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import validation_curve
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
import numpy as np
from array import *


class myClass:
    """
    showDataSetOnPlot:
    first argument, X_train, get an 2-d numpy array. first-d is for x lable on plot and second-d for y lable
    second argument, y_train, get an 1-d numpy array. it used for labled on first argumnts on plot.
    """
    def showDataSetOnPlot(self, x_train, y_train):
        x1 = np.array([[0, 0]])
        y1 = np.array([0])
        for i in range(0, y_train.shape[0]):
            if y_train[i] == 1:
                x1 = np.append(x1, [x_train[i, :]], axis=0)
                y1 = np.append(y1, 1)

        x1 = np.delete(x1, 0, 0)
        y1 = np.delete(y1, 0, 0)
        x_lable1 = x1[:, 0]
        y_lable1 = x1[:, 1]
        plt.scatter(x_lable1, y_lable1, marker='*', color='g')

        x0 = np.array([[0, 0]])
        y0 = np.array([0])
        for i in range(0, y_train.shape[0]):
            if y_train[i] == 0:
                x0 = np.append(x0, [x_train[i, :]], axis=0)
                y0 = np.append(y0, 0)

        x0 = np.delete(x0, 0, 0)
        y0 = np.delete(y0, 0, 0)

        x_lable0 = x0[:, 0]
        y_lable0 = x0[:, 1]
        plt.scatter(x_lable0, y_lable0, marker='^', color='r')
        plt.show()

    """
    showLossCurvePlot:
    first argument must be and estimator from GridSearchCV. method shows the loss curve for estimator.
    x label is number of epochs and y label is loss score for training estimator
    """
    def showLossCurvePlot(self, estimator):
        y_array = estimator.loss_curve_
        x_array = [0]
        for i in range(1, len(y_array)):
            x_array.append(i)
        plt.title("loss curve")
        plt.ylabel('loss')
        plt.xlabel('number of iterations(epochs)')
        plt.plot(x_array, y_array, label="loss score")
        plt.legend(loc="best")
        plt.show()

    # =========================
    # functions1
    def sortData(self, X_train, y_train):
        x = np.array([[0, 0]])
        y = np.array([0])
        for i in range(0, y_train.shape[0]):
            if y_train[i] == 1:
                x = np.append(x, [X_train[i, :]], axis=0)
                # arr1 = np.insert(arr1, i, X_train[i, :])
                y = np.append(y, 1)

        for i in range(0, y_train.shape[0]):
            if y_train[i] == 0:
                x = np.append(x, [X_train[i, :]], axis=0)
                y = np.append(y, 0)

        x = np.delete(x, 0, 0)
        y = np.delete(y, 0, 0)
        return x, y

    def getAllLoss(self, mlp, param_range, param_name, cv=5, X_train=None, y_train=None):
        num = len(param_range)
        list = []
        for i in range(0, num):
            h = param_range[i]
            print(type(h))
            mydict = {param_name: [h]}
            clf = GridSearchCV(estimator=mlp, param_grid=mydict, cv=cv)

            clf.fit(X_train, y_train)
            print(clf.cv_results_)
            list.append(clf.best_estimator_.loss_)
        return list

    def showLossAndMeansOnPlot(self, x, loss, means):
        loss = [float(i) for i in loss]
        means = [float(i) for i in means]
        means = [1 - i for i in means]
        plt.plot(x, loss, label="Training score curve", color="darkorange")
        plt.plot(x, means, label="test score curve", color="blue")
        plt.title("Validation and loss Curve with different hidden layer in mlp")
        plt.ylabel('Score')
        plt.xlabel('hidden layers')
        plt.legend(loc="best")
        plt.show()

    # =========================
    # functions1
    def getSplitTestScore(self, cv_results_, cv):
        text = ''
        sizeOfHyperparam = len(clf.cv_results_['params'])
        myArr = np.zeros((1, sizeOfHyperparam))
        # myArr = np.array([[0, 0, 0]])
        for i in range(0, cv):
            text = 'split' + str(i) + '_test_score'
            myArr = np.append(myArr, [cv_results_[text]], axis=0)
        myArr = np.delete(myArr, 0, 0)
        return myArr

    # =========================
    # plot validation curve
    def showValidationAndLearning_cure_baseOnHiddenLayer(self, mlp, X_train, y_train, param_name='hidden_layer_sizes',
                                                         param_range=None, cv=5):
        train_scores, test_scores = validation_curve(mlp, X_train, y_train, param_name=param_name,
                                                     param_range=param_range,
                                                     scoring="accuracy", cv=cv)
        train_scores = 1 - train_scores
        test_scores = 1 - test_scores
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)

        plt.title("Validation Curve with mlp")
        plt.xlabel("hidden layer")
        plt.ylabel("Score")
        plt.ylim(0.0, 1.1)
        lw = 2
        plt.semilogx(param_range, train_scores_mean, label="Training score",
                     color="darkorange", lw=lw)
        plt.fill_between(param_range, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.2,
                         color="darkorange", lw=lw)

        plt.semilogx(param_range, test_scores_mean, label="Cross-validation score",
                     color="navy", lw=lw)
        plt.fill_between(param_range, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.2,
                         color="navy", lw=lw)
        plt.legend(loc="best")
        plt.show()

    # plt.set_xlabel("Training examples")
    # plt.set_ylabel("Score")

    # =========================
    # Plot learning curve
    def plot_learning_curve(self, estimator, X, y, cv=5):
        train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv)
        train_scores = 1 - train_scores
        test_scores = 1 - test_scores
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        plt.grid()
        plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
        plt.legend(loc="best")
        plt.show()

    # =========================
    # functions1
    def myerror(self, X_test, y_test, clf):
        y_clf = clf.predict(X_test)
        cnt = 0
        print()
        for i in range(0, y_test.size):
            print(str(y_clf[i]) + "," + str(y_test[i]))
            if (y_clf[i] != y_test[i]):
                cnt = cnt + 1

        print("test errors")
        print(cnt / y_test[0].size)

    def showPlt5(self, mydata):
        x_size = mydata.shape[0]
        y_size = mydata.shape[1]

        for j in range(0, y_size):
            x_array = array('f', [])
            y_array = array('f', [])
            for i in range(0, x_size):
                x_array.append(i + 1)
                y_array.append(1 - mydata[i, j])
            plt.title('for number of hidden' + str(j + 1))
            plt.xlabel('number of hidden')
            plt.ylabel('y - axis')
            plt.plot(x_array, y_array)
            plt.show()
