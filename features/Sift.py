from loader.DataSet import load_default
import cv2


class SiftDetector(object):
    def __init__(self, data_set):
        self._data_set = data_set
        self._x_train, self._x_test, self._y_train, self._y_test =\
            self._data_set.x_train, self._data_set.x_test, self._data_set.y_train, self._data_set.y_test
        self._n_train = len(self._y_train)
        self._n_test = len(self._y_test)
        self._detector = cv2.xfeatures2d.SIFT_create()
        self.train_features = None
        self.test_features = None

    def calculate_features_train(self):
        self.train_features = []
        for i in range(self._n_train):
            features = self._detector.detectAndCompute(self._x_train[i, :, :], None)
            self.train_features.append(features)

    def calculate_features_test(self):
        self.test_features = []
        for i in range(self._n_test):
            features = self._detector.detectAndCompute(self._x_test[i, :, :], None)
            self.test_features.append(features)

if __name__== "__main__":
    def main():
        data_set = load_default()
        sift_detector = SiftDetector(data_set)
        sift_detector.calculate_features_train()
        sift_detector.calculate_features_test()
    main()