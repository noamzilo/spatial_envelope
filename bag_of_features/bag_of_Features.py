from loader.DataSet import load_default
from features.Sift import SiftDetector
from sklearn.cluster import KMeans
import numpy as np


class BagOfFratures(object):
    def __init__(self, keppoints_per_image, descriptors_per_image, k=100):
        self._k = k
        self._k_means_obj = KMeans(n_clusters=self._k,
                                   init='k-means++',
                                   n_init=10, # times to run kmeans with different centroid seeds
                                   max_iter=300,  # max iterations per single kmeans run
                                   tol=0.0001,
                                   precompute_distances=True,  # true is faster but takes more memory
                                   verbose=0,
                                   random_state=17,  # random seed
                                   copy_x=True,
                                   n_jobs=-1,  # use max processors possible in parallel
                                   algorithm='auto')
        self._keppoints_per_image = keppoints_per_image
        self._descriptors_per_image = descriptors_per_image
        n_images = len(self._keppoints_per_image)
        assert len(self._descriptors_per_image) == n_images

        self._cluster_histogram_per_image = None

    def calculate_k_means(self):
        flat_descriptors = np.vstack(self._descriptors_per_image)
        self._k_means_obj.fit(flat_descriptors)

    def fit_train_to_centers(self):
        self._cluster_histogram_per_image = []
        for image_descriptors in self._descriptors_per_image:
            centers_ind = self._k_means_obj.predict(image_descriptors)
            histogram = np.histogram(centers_ind, bins=self._k)
            self._cluster_histogram_per_image.append(histogram)


if __name__ == "__main__":
    def main():
        data_set = load_default()
        sift_detector = SiftDetector(data_set)
        sift_detector.calculate_features_train()
        bof = BagOfFratures(sift_detector.train_keypoints, sift_detector.train_descriptors, k=100)
        bof.calculate_k_means()
        bof.fit_train_to_centers()
        

    main()
