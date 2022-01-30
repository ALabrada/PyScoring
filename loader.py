import numpy as np
import os
import csv
import h5py

stages = ['Wake', 'REM', 'N1', 'N2', 'N3']


class DataLoader:
    def __init__(self, dir_path: str, features: str):
        self.dir_path = dir_path
        if features and os.path.exists(features) and os.path.isfile(features):
            with open(features, 'r') as f:
                self.features = [line.rstrip('\n') for line in f.readlines()]
        elif features:
            self.features = str.split(features, sep=',')

    def _load_npz_file(self, npz_file):
        """Load data and labels from a npz file."""
        with np.load(npz_file) as f:
            data = f["x"]
            labels = f["y"]
        return data, labels

    def _load_h5_file(self, h5_file):
        """Load data and labels from a HDF5 file."""
        with h5py.File(h5_file, 'r') as f:
            data = np.array(f.get("x"))
            labels = np.array(f.get("y"))
        return data, labels

    def _load_csv_file(self, csv_file: str):
        """Load data and labels from a CSV file."""
        with open(csv_file, "r") as csvfile:
            datareader = csv.reader(csvfile)
            header = next(datareader)  # yield the header row
            data = []
            labels = []
            for row in datareader:
                labels.append(stages.index(row[0]))
                features = row[1:] if self.features is None else [row[header.index(name)] for name in self.features]
                data.append(np.array(features, dtype=np.float64))
            data = np.vstack(data)
            labels = np.array(labels, dtype=np.int)
        return data, labels

    def load_data(self):
        allfiles = os.listdir(self.dir_path) if os.path.isdir(self.dir_path) else [self.dir_path]
        data = []
        labels = []
        for fname in allfiles:
            print("Loading {} ...".format(fname))
            if ".npz" in fname:
                tmp_data, tmp_labels = self._load_npz_file(fname)
            elif ".csv" in fname:
                tmp_data, tmp_labels = self._load_csv_file(fname)
            elif ".h5" in fname:
                tmp_data, tmp_labels = self._load_h5_file(fname)
            else:
                continue
            data.append(tmp_data)
            labels.append(tmp_labels)
        data = np.vstack(data)
        labels = np.hstack(labels)

        return data, labels

