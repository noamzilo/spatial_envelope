# spatial_envelope

Classify images from 

http://people.csail.mit.edu/torralba/code/spatialenvelope/spatial_envelope_256x256_static_8outdoorcategories.zip

using Bag of Words and SIFT

Full report is under /report/report.docx

# BOW [Bag of Words]

* Choose 2 categories from the above dataset
* Calculate dense-sift for all picrutes
* Vector quantization - Calculate K-Means for all extracted features. Choose at least K=100 means.
* For each image, calculate a histogram of its features of the k'th cluster.
* Train a linear SVM, where each picture is represented by its histogram, and each picture is labeled by its class.

* For the test phase:
* Calculate SIFT for each image.
* For each feature, find its nearest neighbour.
* For each image, calculate its histogram.
* Classify using now trained SVM.
