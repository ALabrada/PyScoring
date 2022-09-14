from argparse import ArgumentParser
from joblib import load
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType, Int64TensorType
from sklearn.ensemble import VotingClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from loader import stages
import os


def export(model_path: str, output_path: str):
    print('Loading model from', model_path)
    classifier = load(model_path)
    features = classifier.n_features_in_ if hasattr(classifier, 'n_features_in_') else None

    print('Converting the classifier to ONNX')
    initial_type = [('features', FloatTensorType([None, features]))]
    final_type = [('label', Int64TensorType([None])), ('probabilities', FloatTensorType([None, len(stages)]))]
    options = {}
    if isinstance(classifier, VotingClassifier):
        options = {
            LinearDiscriminantAnalysis: {'zipmap': False},
            MLPClassifier: {'zipmap': False},
            LinearSVC: {},
        }
        if classifier.voting == 'soft':
            final_type[1] = ('probabilities', FloatTensorType([None, len(classifier.estimators) * len(stages)]))

    onx = convert_sklearn(
        classifier,
        name=os.path.basename(model_path),
        initial_types=initial_type,
        final_types=final_type,
        options=options,
        verbose=2)

    if output_path:
        print('Saving ONNX model to', output_path)
        with open(output_path, "wb") as f:
            f.write(onx.SerializeToString())

    return onx


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--output', type=str, default='model.onnx',
                        help='Path where to save the exported model.')
    parser.add_argument('--model', type=str, required=True,
                        help='File that contains the trained model')
    args = parser.parse_args()
    export(output_path=args.output,
           model_path=args.model)
