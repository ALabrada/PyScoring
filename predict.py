from loader import DataLoader, stages
from sklearn.metrics import confusion_matrix, accuracy_score, cohen_kappa_score, f1_score
from sklearn.model_selection import KFold
from argparse import ArgumentParser
import numpy as np
from joblib import load


def _predict(classifier, data, y_true):
    y_pred = classifier.predict(data)

    cm = confusion_matrix(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    kappa = cohen_kappa_score(y_true, y_pred)

    print(f'acc={acc:.4f}, kappa={kappa:.4f}, f1={f1:.4f}')
    print(cm)
    print(" ")

    return y_pred


def predict(model_path: str, data_path: str, features: str, n_folds: int = 0):
    print('Loading model from', model_path)
    classifier = load(model_path)

    loader = DataLoader(dir_path=data_path,
                        features=features)
    data, labels = loader.load_data()

    if n_folds > 1:
        folds = KFold(n_splits=n_folds)
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
    parser.add_argument('--folds', type=int, default=0,
                        help='Number of folds for cross validation.')
    args = parser.parse_args()
    predict(model_path=args.model,
            data_path=args.data,
            n_folds=args.folds,
            features=args.features)
