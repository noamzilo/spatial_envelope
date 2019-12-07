from sklearn.svm import SVC
from bag_of_features.BagOfFeatures import calculate_bag_of_features_for_default_dataset
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import numpy as np


class Svm(object):
    def __init__(self):
        self._classifier = None

        self._fprs = []
        self._tprs = []
        self._aucs = []

    def train(self, features, labels, c):
        self._classifier = SVC(random_state=0, tol=1e-5, kernel="linear", probability=True, C=c)
        self._classifier.fit(features, labels)

    def predict(self, features):
        predicions = self._classifier.predict(features)
        return predicions

    def calculate_roc(self, test_bag_of_features, test_labels):
        y_score = self._classifier.decision_function(test_bag_of_features)

        fpr, tpr, _ = roc_curve(test_labels, y_score)
        roc_auc = auc(fpr, tpr)

        self._fprs.append(fpr)
        self._tprs.append(tpr)
        self._aucs.append(roc_auc)

    def plot_rocs(self):
        plt.figure()
        # plt.title(f"Roc curve, auc={roc_auc}")
        plt.xlabel('False positive ratio')
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.ylabel('True positive ratio')
        plt.plot(self._fprs, self._tprs)
        plt.show(block=True)


if __name__ == "__main__":
    def main():
        train_bag_of_features, test_bag_of_features, train_labels, test_labels \
            = calculate_bag_of_features_for_default_dataset()

        c_values = np.logspace(-5, 5, 10)
        for c in c_values:
            svm = Svm()
            svm.train(train_bag_of_features, train_labels, c)
            svm.calculate_roc(test_bag_of_features, test_labels)

    main()
