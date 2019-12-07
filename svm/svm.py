from sklearn.svm import LinearSVC
import numpy as np
from bag_of_features.BagOfFeatures import calculate_bag_of_features_for_default_dataset
from sklearn.metrics import auc
from sklearn import metrics

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


class Svm(object):
    def __init__(self, c):
        self._classifier = LinearSVC(random_state=0, tol=1e-5, C=c)

    def train(self, features, labels):
        self._classifier.fit(features, labels)

    def predict(self, features):
        predicions = self._classifier.predict(features)
        return predicions

    def show_roc(self, test_bag_of_features, test_labels):
        metrics.plot_roc_curve(self._classifier, test_bag_of_features, test_labels)
    

def calculate_svm_roc(train_bag_of_features, test_bag_of_features, train_labels, test_labels):
    num_c_tries = 10
    available_cs = np.logspace(start=-2, stop=5.0, num=num_c_tries, base=10.0)
    true_positives = np.zeros(num_c_tries)
    false_positives = np.zeros(num_c_tries)
    # for i, c in enumerate(available_cs):
    #     svm = Svm(c=1)
    #     svm.train(train_bag_of_features, train_labels)
    #     predicions = svm.predict(test_bag_of_features)
    #     confusion = confusion_matrix(test_labels, predicions)
    #     tn, fn, tp, fp = confusion[0, 0], confusion[1, 0], confusion[0, 1], confusion[1, 1]  # true/false positive/negative
    #
    #     true_positives[i] = tp
    #     false_positives[i] = fp
    #
    # return np.array(true_positives)/len(test_labels), np.array(false_positives)/len(test_labels)

    svm = Svm(c=1)
    svm.train(train_bag_of_features, train_labels)
    # predicions = svm.predict(test_bag_of_features)
    svm.show_roc(test_bag_of_features, test_labels)


# def show_roc(true_positives, false_positives):
#     assert true_positives.shape[0] == false_positives.shape[0]
#     roc_auc = auc(fpr, tpr)
#     plt.figure()
#     plt.title("Roc curve")
#     plt.show()

if __name__ == "__main__":
    def main():
        train_bag_of_features, test_bag_of_features, train_labels, test_labels \
            = calculate_bag_of_features_for_default_dataset()

        calculate_svm_roc(train_bag_of_features, test_bag_of_features, train_labels, test_labels)

    main()
