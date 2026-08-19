import pandas as pd
import torch
import torch.nn as nn


def compute_class_weights(train_csv, label_column="Score"):
    df = pd.read_csv(train_csv)

    class_counts = df[label_column].value_counts().sort_index()
    total = class_counts.sum()

    weights = total / (len(class_counts) * class_counts)

    return torch.tensor(weights.values, dtype=torch.float32)


def get_classification_loss(class_weights=None):
    return nn.CrossEntropyLoss(weight=class_weights)


def get_mae_loss():
    return nn.L1Loss()