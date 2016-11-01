#!/usr/bin/python

from csv import reader
from sklearn.feature_selection import SelectKBest


class DataSet:
    def __init__(self, X=None, y=None, col_names=None):
        self._X = X
        self._y = y
        self._column_names = col_names
        self._check_features_sizes()

    def extract_from_csv(self, csv_file):
        self._X = []
        self._y = []
        csv_reader = reader(csv_file, delimiter=',')
        self._column_names = self._read_header(csv_reader)
        for sample in csv_reader:
            self._X.append([float(i) for i in sample[1:]])
            extracted_class = str(sample[0])
            self._y.append(extracted_class)
        self._check_features_sizes()

    def _read_header(self, csv_reader):
        return next(csv_reader)[1:]

    @property
    def X(self):
        return self._X

    @property
    def y(self):
        return self._y

    @property
    def number_of_features(self):
        return len(self.X[0])

    @property
    def col_names(self):
        return self._column_names

    def create_features_ranking(self, use_names=True):
        ranking = []
        for feature_count in range(1, len(self.X[0])+1):
            partial_ranking = SelectKBest(k=feature_count).fit(self.X, self.y).get_support(indices=True)
            for feature_index in partial_ranking:
                if feature_index not in ranking:
                    ranking.append(feature_index)
        if use_names:
            return [self._column_names[index] for index in ranking]
        return ranking

    def _check_features_sizes(self):
        if all([self.X, self.y, self.col_names]):
            prev_len = len(self.X[0])
            for features in self.X[1:]:
                if len(features) != prev_len:
                    raise RuntimeError("Rows sizes mismatch. Check your csv file.")
                prev_len = len(features)
            if prev_len != len(self._column_names):
                raise RuntimeError("Not all columns have names. Check your csv file.")
        elif any([self.X, self.y, self.col_names]):
            raise RuntimeError("Wrong initial data")
