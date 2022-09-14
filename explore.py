from loader import DataLoader, stages
from sklearn.metrics import confusion_matrix, accuracy_score, cohen_kappa_score, f1_score, precision_score, recall_score
from sklearn.model_selection import KFold, StratifiedGroupKFold
from argparse import ArgumentParser
import numpy as np
from joblib import load
import time


def _cmp(x, y):
    return {
        "corr": np.corrcoef(x, y)[0, 1],
        "fit": np.polyfit(x, y, 1).tolist(),
        "min": np.min(x),
        "max": np.max(x),
        "mean": np.mean(x),
        "var": np.var(x)
    }

def _explore(data, labels_values, label_names):
    if not label_names:
        label_names = ["Label"]
        labels_values = labels_values[np.newaxis, :]
    for idx, label_name in enumerate(label_names):
        print("Analyze label {}".format(label_name))
        labels = labels_values[idx, :]

        label_min = np.min(labels)
        label_max = np.max(labels)
        print("Label range: [{}, {}]".format(label_min, label_max))

        results = {c: _cmp(data[c], labels) for c in data.columns}
        results = list(results.items())
        results.sort(key=lambda x: -abs(x[1]["corr"]))
        print("Best correlations:")
        for k, v in results[:5]:
            print("Feature {}: {}".format(k, v))


def explore(data_path: str, features: str, labels_names: list, n_folds: int = 0):
    loader = DataLoader(dir_path=data_path,
                        features=features,
                        labels=labels_names if labels_names else None)
    data, labels, groups = loader.load_data()
    n_features = data.shape[1]

    if n_folds > 1:
        n_groups = groups.nunique()
        if n_groups <= n_folds:
            folds = KFold(n_splits=n_folds)
        else:
            folds = StratifiedGroupKFold(n_splits=n_folds)

        for idx, fold in enumerate(folds.split(data)):
            print("Fold", idx + 1)
            train, test = fold
            _explore(data[test], labels[test], label_names=labels_names)
    else:
        print('Classifier performance:')
        _explore(data, labels, label_names=labels_names)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data', type=str, required=True,
                        help='File or directory that contains the training cases.')
    parser.add_argument('--features', type=str,
                        help='Comma separated list of features or path of a file that contains the features.')
    parser.add_argument('--labels', type=str,
                        help='Comma separated list of label names.')
    parser.add_argument('--folds', type=int, default=0,
                        help='Number of folds for cross validation.')
    args = parser.parse_args()
    explore(data_path=args.data,
            n_folds=args.folds,
            labels_names=args.labels.split(sep=',') if args.labels else None,
            features=args.features)