from loader.DataSet import load_default
from features.Sift import SiftDetector


class BagOfFratures(object):
    def __init__(self, featurs):
        self._featues = featurs


if __name__ == "__main__":
    def main():
        data_set = load_default()
        sift_detector = SiftDetector(data_set)
        bof = BagOfFratures(sift_detector)
        

    main()
