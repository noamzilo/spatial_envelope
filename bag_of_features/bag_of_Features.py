from loader.DataSet import DataSet

class BagOfFratures(object):
    def __init__(self, data_set):
        self._data_set = data_set
        self._x_train, self._x_test, self._y_train, self._y_test =\
            self._data_set.x_train, self._data_set.x_test, self._data_set.y_train, self._data_set.y_test
