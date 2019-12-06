from sklearn.svm import LinearSVC
from loader.DataSet import load_default
from features.Sift import SiftDetector
import numpy as np
from bag_of_features.BagOfFeatures import calculate_bag_of_features_for_default_dataset



class Svm(object):
    def __init__(self):
        self._classifier = LinearSVC(random_state=0, tol=1e-5)

    def train(self, features, labels):
        n_samples = len(features)
        assert features.shape[0] == n_samples
        self._classifier.fit(features, labels)

    def predict_and_evaluate(self, features, labels):
        preds = self._classifier.predict(features)
        correct = np.equal(preds, labels)
        print(f"correct %: {correct}")


if __name__ == "__main__":
    def main():
        data_set = load_default()
        train_bag_of_features, test_bag_of_features = calculate_bag_of_features_for_default_dataset()

        svm = Svm()
        svm.train()

    main()
