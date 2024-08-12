# src/metrics.py

from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
import numpy as np

def accuracy(data_from_opener, predictions):
    y_true = data_from_opener["labels"]

    return accuracy_score(y_true, np.argmax(predictions, axis=1))

def roc_auc(data_from_opener, predictions):
    y_true = data_from_opener["labels"]

    n_class = np.max(y_true) + 1
    y_true_one_hot = np.eye(n_class)[y_true]

    return roc_auc_score(y_true_one_hot, predictions)