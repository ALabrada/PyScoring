import numpy as np
import pandas as pd
import os
import csv
import h5py
from imblearn.under_sampling import RandomUnderSampler

stages = ['Wake', 'N1', 'N2', 'N3', 'REM']


class DataLoader:
    def __init__(self, dir_path: str, features: str = None, balance: bool = False):
        self.dir_path = dir_path
        self.balance = balance
        if features and os.path.exists(features) and os.path.isfile(features):
            with open(features, 'r') as f:
                self.features = [line.rstrip('\n') for line in f.readlines()]
        elif features:
            self.features = str.split(features, sep=',')
        else:
            self.features = None

    def _load_npz_file(self, npz_file):
        """Load data and labels from a npz file."""
        with np.load(npz_file) as f:
            data = f["x"]
            labels = f["y"]
            headers = f["features"]
        frame = pd.DataFrame(data, columns=headers)
        return frame, labels

    def _load_h5_file(self, h5_file):
        """Load data and labels from a HDF5 file."""
        with h5py.File(h5_file, 'r') as f:
            data = np.asarray(f.get("x"), dtype=np.float64)
            # labels = np.asarray([stages.index(y) for y in f.get("y")], dtype=np.int32)
            labels = np.asarray(f.get("y"), dtype=np.str_)
            headers = np.asarray(f.get("features"), dtype=np.str_)
        frame = pd.DataFrame(data, columns=headers)
        labels = pd.Series(data=labels).map({value: idx for idx, value in enumerate(stages)})
        return frame, labels

    def _load_csv_file(self, csv_file: str):
        """Load data and labels from a CSV file."""
        frame: pd.DataFrame = pd.read_csv(csv_file, encoding='utf-8-sig')
        labels = frame.pop('Label')
        labels = labels.map({value: idx for idx, value in enumerate(stages)})
        return frame, labels

    def load_frames(self):
        allfiles = [os.path.join(self.dir_path, x) for x in os.listdir(self.dir_path)] if os.path.isdir(self.dir_path) \
            else [self.dir_path]
        result: pd.DataFrame = None
        for idx, fname in enumerate(allfiles):
            print("Loading {} ...".format(fname))
            if ".npz" in fname:
                data, labels = self._load_npz_file(fname)
            elif ".csv" in fname:
                data, labels = self._load_csv_file(fname)
            elif ".h5" in fname:
                data, labels = self._load_h5_file(fname)
            else:
                continue
            if self.balance:
                data, labels = RandomUnderSampler().fit_resample(data, labels)
            if self.features:
                for c in data.columns:
                    if c not in self.features:
                        data.pop(c)
                assert len(self.features) == data.columns.size, 'Some columns are missing'

            data.insert(0, 'Label', labels)
            data.insert(0, 'Group', idx)
            result = data if result is None else result.append(data, ignore_index=True, sort=False)
        return result

    def load_data(self):
        frame: pd.DataFrame = self.load_frames()
        y = frame.pop('Label')
        g = frame.pop('Group')
        return frame, y, g


