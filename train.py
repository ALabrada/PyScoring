from loader import DataLoader, stages
from sklearn import svm
from sklearn.metrics import make_scorer, accuracy_score, cohen_kappa_score, f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_validate, StratifiedGroupKFold
from imblearn.under_sampling import RandomUnderSampler
from argparse import ArgumentParser
import numpy as np
import math
from joblib import dump


def _print_n_samples_each_class(labels):
    unique_labels = np.unique(labels)
    for c in unique_labels:
        n_samples = len(np.where(labels == c)[0])
        print("{}: {}".format(stages[c], n_samples))


def _create_model(model: str, data, normalize=True):
    if model == 'DT':
        return DecisionTreeClassifier(
            max_depth=20,
            min_samples_leaf=5,
            criterion='entropy',
            class_weight='balanced'
        )
    elif model == 'RF':
        return RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            criterion='entropy',
            class_weight='balanced'
        )
    elif model == 'SVM':
        svc = svm.LinearSVC(
            C=1,
            dual=False,
            max_iter=2000,
            class_weight='balanced')
        return make_pipeline(StandardScaler(), svc) if normalize else svc
    elif model == 'NB':
        return make_pipeline(StandardScaler(), GaussianNB()) if normalize else GaussianNB()
    elif model == 'MLP':
        layers = math.ceil((data.shape[1] + len(stages))/2)
        net = MLPClassifier(hidden_layer_sizes=(layers,),
                            alpha=0.001,
                            max_iter=500)
        return make_pipeline(StandardScaler(), net) if normalize else net
    elif model == 'LDA':
        return LinearDiscriminantAnalysis()
    elif model == 'QDA':
        return QuadraticDiscriminantAnalysis()
    elif model == 'GMM':
        return GaussianMixture()
    elif model == '*':
        vote = VotingClassifier([
            # ('RF', _create_model('RF', data)),
            ('MLP', _create_model('MLP', data, normalize=False)),
            ('LDA', _create_model('LDA', data, normalize=False)),
            ('SVM', _create_model('SVM', data, normalize=False)),
        ], flatten_transform=False)
        return make_pipeline(StandardScaler(), vote)
    else:
        raise Exception(f'Invalid model {model}.')


def train(input_path: str, output_path: str, model_name: str, features: str, balance: bool):
    loader = DataLoader(dir_path=input_path, features=features, balance=balance)
    data, labels, _ = loader.load_data()

    trainer = _create_model(model_name, data)
    classifier = trainer.fit(data, labels)

    if output_path:
        dump(classifier, output_path)
    return classifier


def cross(input_path: str, model_name: str, n_folds: int, features: str, balance: bool):
    loader = DataLoader(dir_path=input_path, features=features, balance=balance)
    data, labels, groups = loader.load_data()
    print('Loaded dataset from', input_path)
    _print_n_samples_each_class(labels)

    trainer = _create_model(model_name, data)
    scorers = {
        'accuracy': make_scorer(accuracy_score),
        'kappa': make_scorer(cohen_kappa_score),
        'f1': make_scorer(lambda x, y: f1_score(x, y, average="macro"))
    }
    n_groups = groups.nunique()
    if n_groups <= n_folds:
        return cross_validate(trainer, data, labels,
                              cv=n_folds,
                              scoring=scorers,
                              verbose=3,
                              n_jobs=4)
    else:
        cv = StratifiedGroupKFold(n_splits=n_folds)
        return cross_validate(trainer, data, labels,
                              cv=cv,
                              scoring=scorers,
                              verbose=3,
                              n_jobs=4)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input', type=str, required=True,
                        help='Input file or directory that contains the training cases.')
    parser.add_argument('--model', type=str, default='*',
                        choices=['DT', 'RF', 'SVM', 'NB', 'MLP', 'LDA', 'QDA', 'GMM', '*'],
                        help='Name of the classifier to build')
    parser.add_argument('--balance', action='store_true',
                        help='Balance the dataset')
    parser.add_argument('--features', type=str,
                        help='Comma separated list of features or path of a file that contains the features.')
    parser.add_argument('--folds', type=int, default=0,
                        help='Number of folds for cross validation.')
    parser.add_argument('--output', type=str,
                        help='Path where to save the trained model ')
    args = parser.parse_args()
    if args.output:
        print('Training classifier')
        train(input_path=args.input,
              output_path=args.output,
              model_name=args.model,
              features=args.features,
              balance=args.balance)
        print('Trained model saved to', args.output)
    if args.folds and args.folds > 1:
        print(f'Performing {args.folds}-fold cross-validation')
        scores = cross(input_path=args.input,
                       model_name=args.model,
                       n_folds=args.folds,
                       features=args.features,
                       balance=args.balance)
        print('Scoring results:')
        for (key, value) in scores.items():
            print(f'{key} = {np.mean(value):.4f} ± {np.std(value):.4f} ({np.min(value):.2f} - {np.max(value):.2f})')

