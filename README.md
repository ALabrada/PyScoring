# PyScoring
Trainer of Machine Learning algorithms for sleep stage scoring. Allows to train scoring algorithms, to perform feature selection, to evaluate the performance and to export trained models to ONNX format.

##Requirements
The application was tested in Python 3.7 with the following packages:

- numpy (1.21.5)
- scikit-learn (1.0.2)
- pandas (0.24.2)
- h5py (2.10.0): Only for reading HDF5 files

##Usage
Select 35 features for a Support Vector Machine (SVM) model from a dataset stored in a CSV file and save them in a text file:

`python fselect.py --input "scoring-features.csv" --model SVM --features 35 --output "features/SVM.txt"`

Train a SVM using 2 epoch sequences with the selected features from the same dataset and save the trained model to SVM.jobs:

`python train.py --input "data\scoring-features.csv" --model SVM --features "features/SVM.txt" --output "models/SVM.jobs" --seq_len 2`

Evaluate the previously trained model using 2 epoch sequences from the same dataset:

`python predict.py --data "data\scoring-features.csv" --model "models/SVM.jobs" --seq_len 2 --features "features/SVM.txt"`

Export the trained model to ONNX:

`python export.py --output models/SVM.onnx --model models/SVM.jobs`
