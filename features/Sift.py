from loader.DataSet import load_default
import cv2
from sklearn.cluster import KMeans


class SiftDetector(object):
    def __init__(self, data_set):
        # https://docs.opencv.org/master/da/df5/tutorial_py_sift_intro.html
        self._data_set = data_set
        self._x_train, self._x_test, self._y_train, self._y_test =\
            self._data_set.x_train, self._data_set.x_test, self._data_set.y_train, self._data_set.y_test
        self._n_train = len(self._y_train)
        self._n_test = len(self._y_test)
        self._detector = cv2.xfeatures2d.SIFT_create()
        self._train_features = None
        self._test_features = None
        self.train_keypoints, self.train_descriptors = None, None
        self.test_keypoints, self.test_descriptors = None, None

    def calculate_features_train(self):
        self._train_features = []
        for i in range(self._n_train):
            keypoints, descriptors = self._detector.detectAndCompute(self._x_train[i, :, :], None)
            self._train_features.append((keypoints, descriptors))
        self.train_keypoints, self.train_descriptors = list(zip(*self._train_features))

    def calculate_features_test(self):
        self._test_features = []
        for i in range(self._n_test):
            keypoints, descriptors = self._detector.detectAndCompute(self._x_test[i, :, :], None)
            self._test_features.append((keypoints, descriptors))
        self.test_keypoints, self.test_descriptors = list(zip(*self._test_features))

if __name__== "__main__":
    def main():
        data_set = load_default()
        sift_detector = SiftDetector(data_set)
        sift_detector.calculate_features_train()
        sift_detector.calculate_features_test()
    main()