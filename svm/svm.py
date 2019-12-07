from sklearn.svm import SVC
from bag_of_features.BagOfFeatures import calculate_bag_of_features_for_default_dataset
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


class Svm(object):
    def __init__(self):
        self._classifier = SVC(random_state=0, tol=1e-5, kernel="linear", probability=True)

    def train(self, features, labels):
        self._classifier.fit(features, labels)

    def predict(self, features):
        predicions = self._classifier.predict(features)
        return predicions

    def show_roc(self, test_bag_of_features, test_labels):
        y_score = self._classifier.decision_function(test_bag_of_features)

        fpr, tpr, _ = roc_curve(test_labels, y_score)
        roc_auc = auc(fpr, tpr)

        plt.figure()
        plt.title(f"Roc curve, auc={roc_auc}")
        plt.xlabel('False positive ratio')
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.ylabel('True positive ratio')
        plt.plot(fpr, tpr)
        plt.show(block=True)


def show_roc(true_positives, false_positives):
    assert true_positives.shape[0] == false_positives.shape[0]
    plt.figure()
    plt.title("Roc curve")
    plt.xlabel('False positive ratio')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.ylabel('True positive ratio')
    plt.scatter(false_positives, true_positives)
    plt.show(block=True)


if __name__ == "__main__":
    def main():
        train_bag_of_features, test_bag_of_features, train_labels, test_labels \
            = calculate_bag_of_features_for_default_dataset()
        svm = Svm()
        svm.train(train_bag_of_features, train_labels)
        svm.show_roc(test_bag_of_features, test_labels)

    main()
