import os
import cv2
import sklearn
import numpy as np


class DataSet(object):
    def __init__(self):
        self._images = None
        self._shape = (256, 256)

    def from_path(self, path):
        image_types = ["mountain", "coast"]
        paths_by_type = {t: [] for t in image_types}
        valid_images = [".jpg",]
        for f in os.listdir(path):
            im_path, ext = os.path.splitext(f)
            if ext.lower() not in valid_images:
                continue
            for typ in paths_by_type:
                if typ in im_path:
                    found_path = os.path.abspath(os.path.join(path, im_path + ext))
                    paths_by_type[typ].append(found_path)
                    break

        images_by_types = {t: [] for t in image_types}
        for typ in paths_by_type:
            for path in paths_by_type[typ]:
                im = cv2.imread(path)
                images_by_types[typ].append(im)

        n_images = np.sum([len(images_by_types[t]) for t in images_by_types])
        images_x = np.zeros((n_images, self._shape[0], self._shape[1]))
        images_y = np.zeros(n_images)
        
        for t in images_by_types:
            images_x[]

        flat_paths = [os.path.abspath(p) for typ in paths_by_type for p in paths_by_type[typ]]
        all_images = [cv2.imread(p) for p in flat_paths if os.path.isfile(p)]
        cv2.imshow("example", all_images[5])
        cv2.waitKey(0)


if __name__ == "__main__":
    def main():
        loader = DataSet()
        path = '../spatial_envelope_256x256_static_8outdoorcategories'
        assert os.path.isdir(path)
        loader.from_path(path)

    main()
