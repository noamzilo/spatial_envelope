from loader.DataSet import load_default
from features.Sift import SiftDetector
from sklearn.cluster import KMeans
import numpy as np


class BagOfFratures(object):
    def __init__(self, k=100):
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

    def calculate_k_means(self, train_descriptors_per_image):
        flat_descriptors = np.vstack(train_descriptors_per_image)
        print("fitting k-means. this takes forever, please wait.")
        self._k_means_obj.fit(flat_descriptors)

    def calculate_cluster_histogram_per_image(self, descriptors_per_image):
        cluster_histogram_per_image = []
        for i, image_descriptors in enumerate(descriptors_per_image):
            centers_ind = self._k_means_obj.predict(image_descriptors)
            histogram, bins = np.histogram(centers_ind, bins=self._k)
            cluster_histogram_per_image.append(histogram)
            if i % 20 == 0:
                print(f"done calculating histogram for image #{i}")
        return cluster_histogram_per_image
    #
    # def predict(self, cluster_histogram_per_image):


def calculate_bag_of_features_for_default_dataset():
    data_set = load_default()
    sift_detector = SiftDetector(data_set)
    sift_detector.calculate_features_train()
    sift_detector.calculate_features_test()
    bof = BagOfFratures(k=100)
    bof.calculate_k_means(sift_detector.train_descriptors)
    train_bag_of_features = bof.calculate_cluster_histogram_per_image(sift_detector.train_descriptors)
    test_bag_of_features = bof.calculate_cluster_histogram_per_image(sift_detector.test_descriptors)

    train_labels, test_labels = data_set.y_train, data_set.y_test
    return train_bag_of_features, test_bag_of_features, train_labels, test_labels

if __name__ == "__main__":
    def main():
        train_bag_of_features, test_bag_of_features, train_labels, test_labels\
            = calculate_bag_of_features_for_default_dataset()

    main()
