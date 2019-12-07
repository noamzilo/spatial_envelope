from sklearn.svm import LinearSVC
import numpy as np
from bag_of_features.BagOfFeatures import calculate_bag_of_features_for_default_dataset
from sklearn.metrics import roc_auc_score


class Svm(object):
    def __init__(self, c):
        self._classifier = LinearSVC(random_state=0, tol=1e-5, C=c)

    def train(self, features, labels):
        self._classifier.fit(features, labels)

    def predict(self, features):
        predicions = self._classifier.predict(features)
        return predicions
    

def calculate_svm_roc(train_bag_of_features, test_bag_of_features, train_labels, test_labels):
    available_cs = np.logspace(start=-2, stop=5.0, num=10, base=10.0)
    # for c in available_cs:
    svm = Svm(c=1)
    svm.train(train_bag_of_features, train_labels)
    predicions = svm.predict(test_bag_of_features)
    auc = roc_auc_score(y_true=labels, y_score=predicions)
    print(f"auc ={auc}")

if __name__ == "__main__":
    def main():
        train_bag_of_features, test_bag_of_features, train_labels, test_labels \
            = calculate_bag_of_features_for_default_dataset()

        calculate_svm_roc(train_bag_of_features, test_bag_of_features, train_labels, test_labels)

    main()
