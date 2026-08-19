import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import wandb

from sklearn.metrics import cohen_kappa_score, confusion_matrix, f1_score


def log_f1_score(all_ground_truth, all_predictions, wb=None, average="weighted", mode="val"):
    y_true = np.asarray(all_ground_truth)
    y_pred = np.asarray(all_predictions)

    f1 = f1_score(y_true, y_pred, average=average)

    print({f"{mode}_f1_score": f1})

    if wb is not None:
        wb.log({f"{mode}_f1_score": f1})

    return f1


def log_confusion_matrix(
    all_ground_truth,
    all_predictions,
    wb,
    class_labels,
    mode="val",
    title="Confusion Matrix",
    normalize=True,
    cmap=plt.cm.Blues,
):
    y_true = np.asarray(all_ground_truth)
    y_pred = np.asarray(all_predictions)

    cm = confusion_matrix(y_true, y_pred, labels=class_labels)
    cm_display = cm.astype(np.float64)

    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)

        with np.errstate(divide="ignore", invalid="ignore"):
            cm_percent = np.divide(
                cm_display,
                row_sums,
                out=np.zeros_like(cm_display),
                where=row_sums != 0,
            ) * 100
    else:
        cm_percent = np.zeros_like(cm_display)

    annotations = np.empty_like(cm, dtype=object)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if normalize:
                annotations[i, j] = f"{cm[i, j]}\n({cm_percent[i, j]:.1f}%)"
            else:
                annotations[i, j] = str(cm[i, j])

    fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(
        cm,
        annot=annotations,
        fmt="",
        cmap=cmap,
        xticklabels=class_labels,
        yticklabels=class_labels,
        ax=ax,
        cbar=normalize,
    )

    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(title)

    plt.tight_layout()

    print(f"{mode}_confusion_matrix:\n{cm}")

    if wb is not None:
        wb.log({f"{mode}_confusion_matrix": wandb.Image(fig)})

    plt.close(fig)

    return cm


def quadratic_weighted_kappa(y_true, y_pred, wb=None, mode="val"):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    kappa = cohen_kappa_score(y_true, y_pred, weights="linear")

    print({f"{mode}_linear_weighted_kappa": kappa})

    if wb is not None:
        wb.log({f"{mode}_linear_weighted_kappa": kappa})

    return kappa