from loader import DataLoader, stages
from sklearn.metrics import confusion_matrix, accuracy_score, cohen_kappa_score, f1_score
from sklearn.model_selection import KFold, StratifiedGroupKFold
from argparse import ArgumentParser
import numpy as np
from joblib import load
import time

def _overlap(data, seq_len=2):
    assert seq_len > 1
    shape = np.shape(data)
    count = shape[0]
    result = np.zeros(shape=(count, seq_len) + shape[1:])
    for start_idx in range(seq_len - 1):
        fragment = data[:start_idx + 1]
        fragment = np.pad(
            fragment,
            pad_width=[(seq_len - start_idx - 1, 0)] + [(0, 0) for _ in shape[1:]],
            mode='edge',
        )
        result[start_idx] = np.reshape(fragment, newshape=(seq_len,) + shape[1:])
    for start_idx in range(seq_len - 1, count):
        result[start_idx] = np.reshape(data[start_idx-seq_len+1:start_idx+1], newshape=(seq_len,) + shape[1:])
    return np.reshape(result, newshape=(count, -1) + shape[2:])


def _predict(classifier, data, y_true):
    start_time = time.time()

    y_pred = classifier.predict(data)

    duration = time.time() - start_time

    cm = confusion_matrix(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    kappa = cohen_kappa_score(y_true, y_pred)

    print(f'acc={acc:.4f}, kappa={kappa:.4f}, f1={f1:.4f}, duration={duration:.3f} s')
    print(cm)
    print(" ")

    return y_pred


def predict(model_path: str, data_path: str, features: str, seq_len: int = 1,  n_folds: int = 0):
    print('Loading model from', model_path)
    classifier = load(model_path)

    loader = DataLoader(dir_path=data_path,
                        features=features)
    data, labels, groups = loader.load_data()
    n_features = data.shape[1]

    if seq_len > 1:
        data = _overlap(data, seq_len=seq_len)

    if n_folds > 1:
        n_groups = groups.nunique()
        if n_groups <= n_folds:
            folds = KFold(n_splits=n_folds)
        else:
            folds = StratifiedGroupKFold(n_splits=n_folds)

        for idx, fold in enumerate(folds.split(data)):
            print("Fold", idx + 1)
            train, test = fold
            _predict(classifier, data[test], labels[test])
    else:
        print('Classifier performance:')
        _predict(classifier, data, labels)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data', type=str, required=True,
                        help='File or directory that contains the training cases.')
    parser.add_argument('--model', type=str, required=True,
                        help='File that contains the trained model')
    parser.add_argument('--features', type=str,
                        help='Comma separated list of features or path of a file that contains the features.')
    parser.add_argument('--seq_len', type=int, default=1,
                        help='')
    parser.add_argument('--folds', type=int, default=0,
                        help='Number of folds for cross validation.')
    args = parser.parse_args()
    predict(model_path=args.model,
            data_path=args.data,
            n_folds=args.folds,
            seq_len=args.seq_len,
            features=args.features)
