from sklearn.svm import SVC
import numpy as np
from bag_of_features.BagOfFeatures import calculate_bag_of_features_for_default_dataset
from sklearn.metrics import auc
from sklearn import metrics

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from tempfile import mkdtemp
from sklearn.externals.joblib import Memory

from sklearn.metrics import roc_curve, auc


class Svm(object):
    def __init__(self, c):
        self._classifier = SVC(random_state=0, tol=1e-5, kernel="linear", probability=True)
        self._c = c

    def train(self, features, labels):
        self._classifier.fit(features, labels)

    def predict(self, features):
        predicions = self._classifier.predict(features)
        return predicions

    def show_roc(self, test_bag_of_features, test_labels):
        # c = self._c
        # w = list(self._classifier.coef_)
        # b = self._classifier.intercept_[0]
        # y_hat = test_bag_of_features.apply(lambda s: np.sum(np.array(s) * np.array(w)) + b, axis=1)
        # y_hat = (y_hat > c)
        # hi=5


        #_____________________
        y_score = self._classifier.decision_function(test_bag_of_features)

        fpr, tpr, _ = roc_curve(test_labels, y_score)
        roc_auc = auc(fpr, tpr)

        fpr["micro"], tpr["micro"], _ = roc_curve(test_labels.ravel(), y_score.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        hi=5

def calculate_svm_roc(train_bag_of_features, test_bag_of_features, train_labels, test_labels):
    num_c_tries = 1000
    # available_cs = np.logspace(start=-5, stop=10.0, num=num_c_tries, base=10.0)
    available_cs = np.linspace(0.00001, 100000, num_c_tries)
    available_cs = 1 / available_cs
    available_cs = np.hstack([available_cs, np.inf])
    true_positive_ratios = np.zeros(num_c_tries)
    false_positive_ratios = np.zeros(num_c_tries)
    for i, c in enumerate(available_cs):
        svm = Svm(c=c)
        svm.train(train_bag_of_features, train_labels)
        predicions = svm.predict(test_bag_of_features)
        confusion = confusion_matrix(test_labels, predicions)
        tn, fn, tp, fp = confusion[0, 0], confusion[1, 0], confusion[0, 1], confusion[1, 1]  # true/false positive/negative

        true_positive_ratios[i] = tp / (tp + fn)
        false_positive_ratios[i] = fp / (fp + tn)

    return np.array(true_positive_ratios), np.array(false_positive_ratios)


def show_roc(true_positives, false_positives):
    assert true_positives.shape[0] == false_positives.shape[0]
    # roc_auc = auc(false_positives, true_positives)
    plt.figure()
    plt.title("Roc curve")
    plt.xlabel('False positive ratio')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.ylabel('True positive ratio')
    plt.scatter(false_positives, true_positives)
    plt.show(block=True)
    # print(f"roc auc: {roc_auc}")


if __name__ == "__main__":
    def main():
        train_bag_of_features, test_bag_of_features, train_labels, test_labels \
            = calculate_bag_of_features_for_default_dataset()
        svm = Svm(c=1)
        svm.train(train_bag_of_features, train_labels)
        svm.show_roc(test_bag_of_features, test_labels)
        # tpr, fpr = calculate_svm_roc(train_bag_of_features, test_bag_of_features, train_labels, test_labels)

    main()
