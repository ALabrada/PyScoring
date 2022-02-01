from loader import DataLoader, stages
from sklearn import svm, naive_bayes
from sklearn.metrics import make_scorer, accuracy_score, cohen_kappa_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import mutual_info_classif, SelectKBest, SelectFromModel, RFE
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import cross_validate
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


def _create_model(model: str, data, max_features: int):
    if model == 'DT':
        dt = DecisionTreeClassifier(
            max_depth=20,
            min_samples_leaf=5,
            criterion='entropy',
            class_weight='balanced'
        )
        return SelectFromModel(dt, prefit=True, max_features=max_features)
    elif model == 'RF':
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            criterion='entropy',
            class_weight='balanced'
        )
        return SelectFromModel(rf, prefit=True, max_features=max_features)
    elif model == 'SVM':
        svc = svm.LinearSVC(C=1, dual=False, class_weight='balanced')
        return make_pipeline(StandardScaler(),
                             SelectFromModel(svc, prefit=True, max_features=max_features))
    elif model == 'NB':
        return make_pipeline(StandardScaler(),
                             SequentialFeatureSelector(naive_bayes(), n_features_to_select=max_features))
    elif model == 'MLP':
        layers = math.ceil((data.shape[1] + len(stages))/2)
        net = MLPClassifier(hidden_layer_sizes=(layers,),
                            alpha=0.001,
                            max_iter=500)
        return make_pipeline(StandardScaler(), SequentialFeatureSelector(net, n_features_to_select=max_features))
    elif model == 'LDA':
        return SelectFromModel(LinearDiscriminantAnalysis(), prefit=True, max_features=max_features)
    elif model == 'QDA':
        return SelectFromModel(QuadraticDiscriminantAnalysis(), prefit=True, max_features=max_features)
    elif model == 'GMM':
        return SelectFromModel(GaussianMixture(), prefit=True, max_features=max_features)
    elif model == 'MI':
        return SelectKBest(mutual_info_classif, k=max_features)
    else:
        raise Exception(f'Invalid model {model}.')


def fselect(input_path: str, output_path: str, model_name: str, max_features: int = 40, balance: bool = False):
    loader = DataLoader(dir_path=input_path)
    data, labels = loader.load_data()

    print('Loaded dataset from', input_path)
    _print_n_samples_each_class(labels)

    if balance:
        data, labels = RandomUnderSampler().fit_resample(data, labels)
        print('Balanced dataset:')
        _print_n_samples_each_class(labels)

    model = _create_model(model_name, data, max_features)
    print('Selecting features...')
    model = model.fit(data, labels)

    features = np.asarray(loader.all_features)[model.get_support()]

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
                        choices=['DT', 'RF', 'SVM', 'NB', 'MLP', 'LDA', 'QDA', 'GMM', 'MI'],
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

