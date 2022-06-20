from loader import DataLoader, stages
from sklearn import svm
from sklearn.base import BaseEstimator, MetaEstimatorMixin
from sklearn.metrics import make_scorer, accuracy_score, cohen_kappa_score, f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.feature_selection import SelectorMixin
from sklearn.feature_selection import mutual_info_classif, SelectKBest, SelectFromModel, RFE, RFECV
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import StratifiedGroupKFold
from argparse import ArgumentParser
import numpy as np
import math
import pandas as pd
import pymrmr


class MRMRSelector(SelectorMixin, MetaEstimatorMixin, BaseEstimator):
    def __init__(self, feature_count: int, mode: str = 'MID'):
        self.mode = mode
        self.feature_count = feature_count

    def fit(self, X, y, **fit_params):
        features = [f'F{i + 1}' for i in range(0, X.shape[1])]
        frame = pd.DataFrame(X, columns=features)
        frame.insert(0, 'Label', y)

        selection = pymrmr.mRMR(frame, self.mode, self.feature_count)
        self.support_ = np.asarray([f in selection for f in features], dtype=np.bool8)

    def _get_support_mask(self):
        return self.support_


def _print_n_samples_each_class(labels):
    unique_labels = np.unique(labels)
    for c in unique_labels:
        n_samples = len(np.where(labels == c)[0])
        print("{}: {}".format(stages[c], n_samples))


def _create_model(model: str, data, max_features: int, n_folds: int = 10, n_groups: int = 0):
    cv = n_folds if n_groups <= 1 else StratifiedGroupKFold(n_splits=min(n_groups, n_folds))
    if model == 'DT':
        dt = DecisionTreeClassifier(
            max_depth=20,
            min_samples_leaf=5,
            criterion='entropy',
            class_weight='balanced'
        )
        return RFECV(
            dt,
            step=5,
            min_features_to_select=max_features,
            cv=cv,
            scoring=make_scorer(lambda x, y: f1_score(x, y, average="macro")),
            n_jobs=4,
            verbose=3
        )
    elif model == 'RF':
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            criterion='entropy',
            class_weight='balanced'
        )
        return RFECV(
            rf,
            step=5,
            min_features_to_select=max_features,
            cv=cv,
            scoring=make_scorer(lambda x, y: f1_score(x, y, average="macro")),
            n_jobs=4,
            verbose=3
        )
    elif model == 'SVM':
        svc = svm.LinearSVC(C=1, dual=False, class_weight='balanced')
        rfe = RFECV(
            svc,
            step=5,
            min_features_to_select=max_features,
            cv=cv,
            scoring=make_scorer(lambda x, y: f1_score(x, y, average="macro")),
            n_jobs=4,
            verbose=3
        )
        return make_pipeline(StandardScaler(), rfe)
    elif model == 'NB':
        sfs = SequentialFeatureSelector(
            GaussianNB(),
            n_features_to_select=max_features,
            cv=cv,
            direction='backward',
            scoring=make_scorer(lambda x, y: f1_score(x, y, average="macro")),
            n_jobs=4
        )
        return make_pipeline(StandardScaler(), sfs)
    elif model == 'MLP':
        layers = math.ceil((data.shape[1] + len(stages))/2)
        net = Perceptron(
            alpha=0.001,
            max_iter=500
        )
        rfe = RFECV(
            net,
            step=2,
            min_features_to_select=max_features,
            cv=cv,
            scoring=make_scorer(lambda x, y: f1_score(x, y, average="macro")),
            n_jobs=4,
            verbose=3
        )
        return make_pipeline(StandardScaler(), rfe)
    elif model == 'LDA':
        return RFECV(
            LinearDiscriminantAnalysis(),
            step=1,
            min_features_to_select=max_features,
            cv=cv,
            scoring=make_scorer(lambda x, y: f1_score(x, y, average="macro")),
            n_jobs=4,
            verbose=3
        )
    elif model == 'QDA':
        return RFECV(
            QuadraticDiscriminantAnalysis(),
            step=5,
            min_features_to_select=max_features,
            cv=cv,
            scoring=make_scorer(lambda x, y: f1_score(x, y, average="macro")),
            n_jobs=4,
            verbose=3
        )
    elif model == 'GMM':
        sfs = SequentialFeatureSelector(
            GaussianMixture(),
            n_features_to_select=max_features,
            cv=cv,
            direction='backward',
            scoring=make_scorer(lambda x, y: f1_score(x, y, average="macro")),
            n_jobs=4
        )
        return make_pipeline(StandardScaler(), sfs)
    elif model == 'MI':
        return SelectKBest(mutual_info_classif, k=max_features)
    elif model == 'mRMR':
        return make_pipeline(StandardScaler(), MRMRSelector(feature_count=max_features))
    else:
        raise Exception(f'Invalid model {model}.')


def fselect(input_path: str, output_path: str, model_name: str, max_features: int = 40, balance: bool = False):
    loader = DataLoader(dir_path=input_path, balance=balance)
    data = loader.load_frames()
    labels = data.pop('Label')

    print('Loaded dataset from', input_path)
    _print_n_samples_each_class(labels)

    groups = data.pop('Group')

    n_groups = groups.nunique()
    model = _create_model(model_name, data, max_features, n_groups=n_groups)
    print('Selecting features...')

    model = model.fit(data, labels, groups=groups) if n_groups > 1 else model.fit(data, labels)

    if isinstance(model, Pipeline):
        mask = model.steps[-1][1].get_support()
    else:
        mask = model.get_support()

    features = np.asarray(data.columns)[mask]

    if output_path:
        print('Saving selected features to', output_path)
        with open(output_path, mode='w') as f:
            f.writelines([f'{f}\n' for f in features])

    return features


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input', type=str, required=True,
                        help='Input file or directory that contains the training cases.')
    parser.add_argument('--model', type=str, default='RF',
                        choices=['DT', 'RF', 'SVM', 'NB', 'MLP', 'LDA', 'QDA', 'GMM', 'MI', 'mRMR'],
                        help='Name of the classifier to build')
    parser.add_argument('--features', type=int, default=40,
                        help='Maximum number of features to select')
    parser.add_argument('--balance', action='store_true',
                        help='Balance the dataset')
    parser.add_argument('--output', type=str,
                        help='Path where to save the trained model ')
    args = parser.parse_args()
    print('Selecting features')
    features = fselect(input_path=args.input,
                       output_path=args.output,
                       model_name=args.model,
                       max_features=args.features,
                       balance=args.balance)
    print('Selected features:')
    for idx, f in enumerate(features):
        print(f'{idx + 1}: \t{f}')

