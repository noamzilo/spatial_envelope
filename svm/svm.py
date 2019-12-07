from sklearn.svm import SVC
from bag_of_features.BagOfFeatures import calculate_bag_of_features_for_default_dataset
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import numpy as np


class Svm(object):
    def __init__(self):
        self._classifier = None

        self._cs = []
        self._fprs = []
        self._tprs = []
        self._aucs = []

    def train(self, features, labels, c):
        self._classifier = SVC(random_state=0, tol=1e-5, kernel="linear", probability=True, C=c)
        self._classifier.fit(features, labels)
        self._cs.append(c)

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
        plt.xlabel('False positive ratio')
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.ylabel('True positive ratio')
        for fpr, tpr, c, auc_ in zip(self._fprs, self._tprs, self._cs, self._aucs):
            if c < 1:
                plt.plot(fpr, tpr, label=f"c={c:.5f}, auc={auc_:.4f}")
            else:
                plt.plot(fpr, tpr, label=f"c={c:.0f}, auc={auc_:.4f}")
        plt.legend(loc="bottom right")
        plt.show(block=True)


if __name__ == "__main__":
    def main():
        train_bag_of_features, test_bag_of_features, train_labels, test_labels \
            = calculate_bag_of_features_for_default_dataset()

        c_values = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5]
        svm = Svm()
        for c in c_values:
            svm.train(train_bag_of_features, train_labels, c)
            svm.calculate_roc(test_bag_of_features, test_labels)

        svm.plot_rocs()
    main()
