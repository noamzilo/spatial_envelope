from loader.DataSet import load_default
from features.Sift import SiftDetector
from sklearn.cluster import KMeans


class BagOfFratures(object):
    def __init__(self, keppoints_per_image, descriptors_per_image):
        self._keppoints_per_image = keppoints_per_image
        self._descriptors_per_image = descriptors_per_image
        n_images = len(self._keppoints_per_image)
        assert len(self._descriptors_per_image) == n_images

        self._k_means_obj = None

        hi=5

    def calculate_k_means(self, k=100):
        self._k_means_obj = KMeans(n_clusters=k,
                                   init='k-means++',
                                   n_init=10, # times to run kmeans with different centroid seeds
                                   max_iter=300, # max iterations per single kmeans run
                                   tol=0.0001,
                                   precompute_distances=True,  # true is faster ut takes more memory
                                   verbose=0,
                                   random_state=17,  # random seed
                                   copy_x=True,
                                   n_jobs=-1,  # use max processors possible in parallel
                                   algorithm='auto')


if __name__ == "__main__":
    def main():
        data_set = load_default()
        sift_detector = SiftDetector(data_set)
        sift_detector.calculate_features_train()
        bof = BagOfFratures(sift_detector.train_keypoints, sift_detector.train_descriptors)
        bof.calculate_k_means(k=100)
        

    main()
