from loader import DataLoader, stages
from sklearn import svm, naive_bayes
from sklearn.metrics import make_scorer, accuracy_score, cohen_kappa_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_validate
from argparse import ArgumentParser
import numpy as np
import math
from joblib import dump


def _print_n_samples_each_class(labels):
    unique_labels = np.unique(labels)
    for c in unique_labels:
        n_samples = len(np.where(labels == c)[0])
        print("{}: {}".format(stages[c], n_samples))


def _create_model(model: str, data):
    if model == 'RF':
        return RandomForestClassifier(n_estimators=100, min_samples_leaf=2)
    elif model == 'SVM':
        return svm.SVC(gamma=0.001, C=100.)
    elif model == 'NB':
        return naive_bayes()
    elif model == 'MLP':
        layers = math.ceil((data.shape[1] + len(stages))/2)
        return MLPClassifier(hidden_layer_sizes=(layers,),
                             learning_rate_init=0.3,
                             max_iter=100,
                             momentum=0.2)
    elif model == 'LDA':
        return LinearDiscriminantAnalysis()
    else:
        raise Exception(f'Invalid model {model}.')


def train(input_path: str, output_path: str, model_name: str):
    loader = DataLoader(dir_path=input_path)
    data, labels = loader.load_data()

    trainer = _create_model(model_name, data)
    classifier = trainer.fit(data, labels)

    if output_path:
        dump(classifier, output_path)
    return classifier


def cross(input_path: str, model_name: str, n_folds: int):
    loader = DataLoader(dir_path=input_path)
    data, labels = loader.load_data()
    print('Loaded dataset from', input_path)
    _print_n_samples_each_class(labels)

    trainer = _create_model(model_name, data)
    scorers = {
        'accuracy': make_scorer(accuracy_score),
        'kappa': make_scorer(cohen_kappa_score),
        'f1': make_scorer(lambda x, y: f1_score(x, y, average="macro"))
    }
    return cross_validate(trainer, data, labels,
                          cv=n_folds,
                          scoring=scorers,
                          verbose=2,
                          n_jobs=4)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input', type=str, required=True,
                        help='Input file or directory that contains the training cases.')
    parser.add_argument('--model', type=str, default='RF', choices=['RF', 'SVM', 'NB', 'MLP', 'LDA'],
                        help='Name of the classifier to build')
    parser.add_argument('--folds', type=int, default=0,
                        help='Number of folds for cross validation.')
    parser.add_argument('--output', type=str,
                        help='Path where to save the trained model ')
    args = parser.parse_args()
    if args.output:
        print('Training classifier')
        train(input_path=args.input,
              output_path=args.output,
              model_name=args.model)
        print('Trained model saved to', args.output)
    if args.folds and args.folds > 1:
        print(f'Performing {args.folds}-fold cross-validation')
        scores = cross(input_path=args.input,
                       model_name=args.model,
                       n_folds=args.folds)
        print('Scoring results:')
        for (key, value) in scores.items():
            print(f'{key} = {np.mean(value):.4f} ± {np.std(value):.4f} ({np.min(value):.2f} - {np.max(value):.2f})')

