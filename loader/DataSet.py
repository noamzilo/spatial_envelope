import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split


class DataSet(object):
    def __init__(self):
        self._images = None
        self._shape = (256, 256)
        self._images_x = None
        self._images_y = None
        self._image_types = ["mountain", "coast"]
        self._image_tags = [i for i in range(len(self._image_types))]
        self.x_train, self.x_test, self.y_train, self.y_test = None, None, None, None

        self._max_images_used = 100

    def load_from_path(self, path):
        paths_by_type = {t: [] for t in self._image_types}
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
        all_images_tagged = [(cv2.imread(p, 0), t) for p, t in flat_paths if os.path.isfile(p)]

        self._images_x = np.array([datum[0] for datum in all_images_tagged])
        self._images_y = np.array([datum[1] for datum in all_images_tagged])

        self._images_x = self._images_x[:self._max_images_used]
        self._images_y = self._images_y[:self._max_images_used]

    def split(self, test_size=0.2):
        x = self._images_x
        y = self._images_y
        self.x_train, self.x_test, self.y_train, self.y_test = \
            train_test_split(x, y, test_size=test_size, random_state=42)


def load_default():
    data_Set = DataSet()
    path = '../spatial_envelope_256x256_static_8outdoorcategories'
    assert os.path.isdir(path)
    data_Set.load_from_path(path)
    data_Set.split(test_size=0.2)
    return data_Set

if __name__ == "__main__":
    def main():
        data_Set = load_default

    main()
