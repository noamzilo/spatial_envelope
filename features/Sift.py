from loader.DataSet import load_default
import cv2


class SiftDetector(object):
    def __init__(self, data_set, num_features_per_image = 200):
        self._data_set = data_set
        self._x_train, self._x_test, self._y_train, self._y_test =\
            self._data_set.x_train, self._data_set.x_test, self._data_set.y_train, self._data_set.y_test
        self._n_train = len(self._y_train)
        self._detector = cv2.xfeatures2d.SIFT_create()

    def calculate_features(self):
        # retval = cv.xfeatures2d.SIFT_create([,nfeatures[,nOctaveLayers[,contrastThreshold[,edgeThreshold[,sigma]]]]])
        train_features = []
        for i in range(self._n_train):
            features = self._detector.detect(self._data_set.x_train[i, :, :], None)
            train_features.append(features)
        hi=5

if __name__== "__main__":
    def main():
        data_set = load_default()
        sift_detector = SiftDetector(data_set)
        sift_detector.calculate_features()
    main()