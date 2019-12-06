from loader.DataSet import load_default
from features.Sift import SiftDetector


class BagOfFratures(object):
    def __init__(self, keppoints_per_image, descriptors_per_image):
        self._keppoints_per_image = keppoints_per_image
        self._descriptors_per_image = descriptors_per_image

        hi=5

    def calculate_k_means(self, k=100):
        print(k)


if __name__ == "__main__":
    def main():
        data_set = load_default()
        sift_detector = SiftDetector(data_set)
        sift_detector.calculate_features_train()
        bof = BagOfFratures(sift_detector.train_keypoints, sift_detector.train_descriptors)
        bof.calculate_k_means(k=100)
        

    main()
