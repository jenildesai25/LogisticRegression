import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_curve, auc
from scipy import interp
import json
from itertools import cycle
from sklearn import svm
from sklearn.model_selection import StratifiedKFold


class LogisticRegression:

    def __init__(self, lr=0.01, num_iter=10000, fit_intercept=True, verbose=True, weight=None, task=None):
        self.lr = lr
        self.num_iter = num_iter
        self.fit_intercept = fit_intercept
        self.verbose = verbose
        self.weight = weight
        self.edge = []
        self.task = task

    @classmethod
    def generate_test_data(cls):
        mu_1 = [1, 0]
        mu_2 = [0, 1.5]

        sigma_1 = [[1, 0.75], [0.75, 1]]
        sigma_2 = [[1, 0.75], [0.75, 1]]
        set_0 = pd.DataFrame(np.append(np.random.multivariate_normal(mu_1, sigma_1, 1000), np.zeros((1000, 1)), axis=1))
        set_1 = pd.DataFrame(np.append(np.random.multivariate_normal(mu_2, sigma_2, 1000), np.ones((1000, 1)), axis=1))
        return set_0, set_1, set_0[2], set_1[2]

    def __add_intercept(self, X):
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)

    def __sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def __loss(self, h, y):
        return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()

    def fit(self, X, y):
        if self.fit_intercept:
            X = self.__add_intercept(X)
        if not self.weight:

            # weights initialization
            self.theta = np.zeros(X.shape[1])
        else:

            self.theta = self.weight
        if self.task == 'task_1':
            for i in range(self.num_iter):
                # if i <= 10000:
                z = np.dot(X, self.theta)
                h = self.__sigmoid(z)
                gradient = np.dot(X.T, (h - y)) / y.size
                self.edge.append(gradient)
                self.theta -= self.lr * gradient

                z = np.dot(X, self.theta)
                h = self.__sigmoid(z)
                loss = self.__loss(h, y)

                if self.verbose and i % 10000 == 0:
                    # print(f'loss: {loss} \t')
                    pass
            # print('weight edges: {}'.format(self.edge))
        else:
            for i in range(self.num_iter):
                z = np.dot(X, self.theta)
                h = self.__sigmoid(z)
                gradient = np.dot(X.T, (h - y)) / y.size
                self.edge.append(gradient)
                self.theta -= self.lr * gradient

                z = np.dot(X, self.theta)
                h = self.__sigmoid(z)
                loss = self.__loss(h, y)

                if self.verbose and i % 10000 == 0:
                    # print(f'loss: {loss} \t')
                    pass
            # print('weight edges: {}'.format(self.edge))
        return self.edge

    def predict_prob(self, X):
        if self.fit_intercept:
            X = self.__add_intercept(X)

        return self.__sigmoid(np.dot(X, self.theta))

    def predict(self, X):
        return self.predict_prob(X).round()


if __name__ == '__main__':
    try:
        user_input = input('Please enter Task_1 to run 1st task or Task_2 to run 2nd task ot Task_3 for 3rd task.Do not enter anything else: ')
        if user_input.lower() == 'task_1':
            n = [1, 0.1, 0.01]
            for i in n:
                logistic_regression_object = LogisticRegression(lr=i)
                set_0_data, set_1_data, set_0_data_labels, set_1_data_labels = LogisticRegression.generate_test_data()
                test_data_frames = [set_0_data[:500], set_1_data[:500]]
                test_data = pd.concat(test_data_frames, ignore_index=True)
                test_data = pd.DataFrame(test_data)
                test_data_labels = test_data[2]
                test_data = test_data.iloc[:, :2]
                edges = logistic_regression_object.fit(test_data, test_data_labels)
                f = open('Task_1.txt', 'w')
                for line in edges:
                    f.write(str(line))
                prediction = logistic_regression_object.predict(test_data)
                accuracy = 0
                rounded_label = [round(elem, 1) for elem in test_data_labels]
                for label_from_train_data, label_from_prediction in zip(rounded_label, prediction):
                    if label_from_train_data == label_from_prediction:
                        accuracy = accuracy + 1
                my_accuracy = 100 * accuracy
                result = my_accuracy / len(rounded_label)
                print('Accuracy of {} {} learning rate is :{} '.format(user_input, i, result))
        elif user_input.lower() == 'task_2':
            n = [1, 0.1, 0.01]
            for i in n:
                logistic_regression_object = LogisticRegression(lr=i, num_iter=10000, task=user_input.lower())
                set_0_data, set_1_data, set_0_data_labels, set_1_data_labels = LogisticRegression.generate_test_data()
                test_data_frames = [set_0_data[:500], set_1_data[:500]]
                test_data = pd.concat(test_data_frames, ignore_index=True)
                test_data = pd.DataFrame(test_data)
                test_data_labels = test_data[2]
                test_data = test_data.iloc[:, :2]
                edges = logistic_regression_object.fit(test_data, test_data_labels)
                # print(edges)
                f = open('Task_2.txt', 'w')
                for line in edges:
                    f.write(str(line))
                prediction = logistic_regression_object.predict(test_data)
                accuracy = 0
                rounded_label = [round(elem, 1) for elem in test_data_labels]
                for label_from_train_data, label_from_prediction in zip(rounded_label, prediction):
                    if label_from_train_data == label_from_prediction:
                        accuracy = accuracy + 1
                my_accuracy = 100 * accuracy
                result = my_accuracy / len(rounded_label)
                print('Accuracy of {} {} learning rate is :{} '.format(user_input, i, result))
        elif user_input.lower() == 'task_3':
            set_0_data, set_1_data, set_0_data_labels, set_1_data_labels = LogisticRegression.generate_test_data()
            test_data_frames = [set_0_data[:500], set_1_data[:500]]
            test_data = pd.concat(test_data_frames, ignore_index=True)
            test_data = pd.DataFrame(test_data)
            y = test_data[2]
            X = test_data.iloc[:, :2]
            X, y = X[y != 2], y[y != 2]
            n_samples, n_features = X.shape
            random_state = np.random.RandomState(0)
            X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]

            cv = StratifiedKFold(n_splits=6)
            classifier = svm.SVC(kernel='linear', probability=True,
                                 random_state=random_state)

            tprs = []
            aucs = []
            mean_fpr = np.linspace(0, 1, 100)

            i = 0
            for train, test in cv.split(X, y):
                probas_ = classifier.fit(X[train], y[train]).predict_proba(X[test])
                # Compute ROC curve and area the curve
                fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
                tprs.append(interp(mean_fpr, fpr, tpr))
                tprs[-1][0] = 0.0
                roc_auc = auc(fpr, tpr)
                aucs.append(roc_auc)
                plt.plot(fpr, tpr, lw=1, alpha=0.3,
                         label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

                i += 1
            plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
                     label='Chance', alpha=.8)

            mean_tpr = np.mean(tprs, axis=0)
            mean_tpr[-1] = 1.0
            mean_auc = auc(mean_fpr, mean_tpr)
            std_auc = np.std(aucs)
            plt.plot(mean_fpr, mean_tpr, color='b',
                     label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
                     lw=2, alpha=.8)

            std_tpr = np.std(tprs, axis=0)
            tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
            tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
            plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                             label=r'$\pm$ 1 std. dev.')

            plt.xlim([-0.05, 1.05])
            plt.ylim([-0.05, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver operating characteristic example')
            plt.legend(loc="lower right")
            plt.show()
        else:
            print('Please enter Task_1 or Task_2 or Task_3.Do not enter anything else.')
    except Exception as e:
        print(e)
