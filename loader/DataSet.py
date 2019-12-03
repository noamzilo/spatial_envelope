import os
import cv2
import sklearn
import numpy as np


class DataSet(object):
    def __init__(self):
        self._images = None
        self._shape = (256, 256)
        self._images_x = None
        self._images_y = None

    def from_path(self, path):
        image_types = ["mountain", "coast"]
        # image_tags = [i for i in range(len(image_types))]
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

        flat_paths = [(os.path.abspath(p), tag) for tag, typ in enumerate(paths_by_type) for p in paths_by_type[typ]]
        all_images = [(cv2.imread(p), t) for p, t in flat_paths if os.path.isfile(p)]

        self._images_x = np.array([datum[0] for datum in all_images])
        self._images_y = np.array([datum[1] for datum in all_images])


if __name__ == "__main__":
    def main():
        loader = DataSet()
        path = '../spatial_envelope_256x256_static_8outdoorcategories'
        assert os.path.isdir(path)
        loader.from_path(path)

    main()
