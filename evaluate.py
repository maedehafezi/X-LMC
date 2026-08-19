import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import torch

from utils.loss import get_classification_loss, get_mae_loss
from utils.metrics import (
    log_confusion_matrix,
    log_f1_score,
    quadratic_weighted_kappa,
)


def evaluate(
    net,
    dataloader,
    device=torch.device("cuda"),
    mode="val",
    wb=None,
    save_predictions=False,
    output_file=None,
):
    if len(dataloader) == 0:
        raise ValueError("Empty validation/test set.")

    net.eval()

    criterion = get_classification_loss()
    criterion_mae = get_mae_loss()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    all_predictions = []
    all_ground_truth = []
    all_file_names = []
    all_patient_ids = []

    with torch.no_grad():
        for images, images2, scores_true, (ap_file, lat_file), patient_ids in dataloader:
            images = images.to(device, dtype=torch.float32)
            images2 = images2.to(device, dtype=torch.float32)
            scores_true = scores_true.to(device, dtype=torch.long)

            scores_pred = net(images, images2)

            loss_ce = criterion(scores_pred, scores_true)

            num_classes = scores_pred.shape[1]
            probabilities = torch.softmax(scores_pred, dim=1)

            targets_one_hot = torch.nn.functional.one_hot(
                scores_true,
                num_classes=num_classes,
            ).float()

            loss_mae = criterion_mae(probabilities, targets_one_hot)

            alpha = 0.3
            loss = (1 - alpha) * loss_ce + alpha * loss_mae

            batch_size = scores_true.size(0)
            predicted_classes = torch.argmax(scores_pred, dim=1)

            total_loss += loss.item() * batch_size
            total_correct += (predicted_classes == scores_true).sum().item()
            total_samples += batch_size

            all_predictions.extend(predicted_classes.cpu().tolist())
            all_ground_truth.extend(scores_true.cpu().tolist())

            if isinstance(ap_file, (list, tuple)) and isinstance(lat_file, (list, tuple)):
                all_file_names.extend(list(zip(ap_file, lat_file)))
            else:
                all_file_names.append((ap_file, lat_file))

            if torch.is_tensor(patient_ids):
                all_patient_ids.extend(patient_ids.cpu().tolist())
            elif isinstance(patient_ids, (list, tuple)):
                all_patient_ids.extend(patient_ids)
            else:
                all_patient_ids.append(patient_ids)

    val_loss = total_loss / total_samples
    val_accuracy = total_correct / total_samples

    labels = list(range(num_classes))

    f1 = log_f1_score(
        all_ground_truth,
        all_predictions,
        wb,
        average="weighted",
        mode=mode,
    )

    log_confusion_matrix(
        all_ground_truth,
        all_predictions,
        wb,
        class_labels=labels,
        mode=mode,
        title=f"{mode.capitalize()} Confusion Matrix",
        normalize=True,
        cmap=plt.cm.Blues,
    )

    qwk = quadratic_weighted_kappa(
        all_ground_truth,
        all_predictions,
        wb,
        mode=mode,
    )

    if save_predictions and output_file:
        df_predictions = pd.DataFrame({
            "Patient ID": all_patient_ids,
            "File Name": all_file_names,
            "Ground Truth": all_ground_truth,
            "Predicted Scores": all_predictions,
        })

        df_predictions.to_csv(output_file, index=False)
        print(f"Predictions saved to {output_file}")

    if wb is not None:
        wb.log({
            f"{mode}_loss": val_loss,
            f"{mode}_accuracy": val_accuracy,
        })

    net.train()

    return val_loss, val_accuracy, f1, qwk