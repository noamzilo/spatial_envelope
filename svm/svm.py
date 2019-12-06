from sklearn.svm import LinearSVC
from loader.DataSet import load_default
from features.Sift import SiftDetector
import numpy as np
from bag_of_features.BagOfFeatures import BagOfFratures


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
        sift_detector = SiftDetector(data_set)
        sift_detector.calculate_features_train()
        train_bof = BagOfFratures(sift_detector.train_keypoints, sift_detector.train_descriptors, k=100)
        train_bof.calculate_k_means()
        train_bof.fit_train_to_centers()

        train_svm = Svm()
    main()