from argparse import ArgumentParser
from joblib import load
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType


def export(model_path: str, output_path: str):
    print('Loading model from', model_path)
    classifier = load(model_path)
    features = classifier.n_features_in_ if hasattr(classifier, 'n_features_in_') else None

    print('Converting the classifier to ONNX')
    initial_type = [('float_input', FloatTensorType([None, features]))]
    onx = convert_sklearn(classifier, initial_types=initial_type, verbose=1)

    print('Saving ONNX model to', output_path)
    with open(output_path, "wb") as f:
        f.write(onx.SerializeToString())


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--output', type=str, default='model.onnx',
                        help='Path where to save the exported model.')
    parser.add_argument('--model', type=str, required=True,
                        help='File that contains the trained model')
    args = parser.parse_args()
    export(output_path=args.output,
           model_path=args.model)
