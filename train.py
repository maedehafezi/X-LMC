import argparse
import logging
import os
import sys
from pathlib import Path

import albumentations as A
import cv2
import pandas as pd
import torch
import torch.nn as nn
import wandb

from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from evaluate import evaluate
from model.xlmc import XLMC
from utils.data_loading import BiplaneDSADataset
from utils.early_stopping import EarlyStopping
from utils.loss import compute_class_weights, get_classification_loss, get_mae_loss
from utils.optimizers import get_optimizer


def train_net(net, epochs, amp, device, wandb_logging, output_checkpoint=None):
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp and device.type == "cuda")

    class_weights = compute_class_weights(args.train_csv).to(device)

    criterion = get_classification_loss(class_weights=class_weights)
    criterion_mae = get_mae_loss()

    early_stopping = EarlyStopping(patience=50, verbose=True, mode="min")

    optimizer = get_optimizer(
        args.optimizer,
        net,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=20,
        factor=0.5,
        min_lr=1e-8,
    )

    for epoch in range(epochs):
        net.train()

        running_loss = 0.0
        running_correct = 0
        running_total = 0

        with tqdm(total=len(train_loader), desc=f"Epoch {epoch + 1}/{epochs}", unit="batch") as pbar:

            for images, images2, scores_true, file_info, patient_ids in train_loader:
                images = images.to(device, dtype=torch.float32)
                images2 = images2.to(device, dtype=torch.float32)
                scores_true = scores_true.to(device, dtype=torch.long)

                optimizer.zero_grad(set_to_none=True)

                with torch.cuda.amp.autocast(enabled=amp and device.type == "cuda"):
                    logits = net(images, images2)

                    loss_ce = criterion(logits, scores_true)

                    probabilities = torch.softmax(logits, dim=1)

                    targets_one_hot = torch.nn.functional.one_hot(
                        scores_true,
                        num_classes=logits.shape[1],
                    ).float()

                    loss_mae = criterion_mae(probabilities, targets_one_hot)

                    alpha = 0.3
                    loss = (1 - alpha) * loss_ce + alpha * loss_mae

                batch_size = scores_true.size(0)

                running_loss += loss.item() * batch_size

                predicted_classes = torch.argmax(probabilities, dim=1)
                running_correct += (predicted_classes == scores_true).sum().item()
                running_total += batch_size

                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(1)
                pbar.set_postfix({"train loss": loss.item()})

                if wandb_logging:
                    wandb_logging.log({
                        "batch_loss": loss.item(),
                        "epoch": epoch,
                    })

            epoch_loss = running_loss / running_total
            epoch_accuracy = running_correct / running_total

            val_loss, val_accuracy, val_f1, val_qwk = evaluate(
                net,
                val_loader,
                task="classification",
                device=device,
                mode="val",
                wb=wandb,
                focal=False,
            )

            if wandb_logging:
                wandb_logging.log({
                    "epoch": epoch,
                    "train_loss": epoch_loss,
                    "train_accuracy": epoch_accuracy,
                    "val_loss": val_loss,
                    "val_accuracy": val_accuracy,
                    "val_f1": val_f1,
                    "val_qwk": val_qwk,
                })

            pbar_postfix_str = (
                f"train loss={epoch_loss:.6f}, "
                f"train accuracy={epoch_accuracy:.4f}, "
                f"val loss={val_loss:.6f}, "
                f"val accuracy={val_accuracy:.4f}, "
                f"val F1={val_f1:.4f}, "
                f"val QWK={val_qwk:.4f}"
            )

            early_stopping(val_loss)

            if early_stopping.save_model:
                model_state_dict = net.module.state_dict() if isinstance(net, nn.DataParallel) else net.state_dict()

                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model_state_dict,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": val_loss,
                }, output_checkpoint)

                pbar_postfix_str += " Saved best model based on val_loss."

            if early_stopping.counter != 0:
                pbar_postfix_str += (
                    f" EarlyStopping counter: {early_stopping.counter}/{early_stopping.patience}"
                )

            pbar.set_postfix_str(pbar_postfix_str)

            if early_stopping.early_stop:
                logging.info("Early stopping")
                break

            current_lr = optimizer.param_groups[0]["lr"]

            if wandb_logging:
                wandb_logging.log({
                    "learning_rate": current_lr,
                    "epoch": epoch,
                })

            scheduler.step(val_loss)

    torch.save(
        torch.load(output_checkpoint, map_location=device),
        os.path.join(wandb_logging.dir, "checkpoint.pt"),
    )


def get_args():
    parser = argparse.ArgumentParser(description="Train X-LMC for collateral score classification.")

    parser.add_argument("--epochs", "-e", type=int, default=1000)
    parser.add_argument("--input-type", "-i", choices=["sequence", "minip"], default="sequence")
    parser.add_argument("--batch-size", "-b", type=int, default=1)
    parser.add_argument("--learning-rate", "-l", type=float, default=1e-3, dest="lr")
    parser.add_argument("--output_checkpoint", "-o", type=str, default=None)
    parser.add_argument("--wandb_project", "-w", type=str, default="CollateralScoring")
    parser.add_argument("--amp", action="store_true")

    parser.add_argument(
        "--optimizer",
        type=str,
        default="adam",
        choices=["adam", "adamw", "rmsprop", "sgd", "ranger"],
    )

    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--img_scale", "-s", type=float, default=224 / 512)

    parser.add_argument("--train_csv", type=str, default="train_M1_fold0.csv")
    parser.add_argument("--val_csv", type=str, default="val_M1_fold0.csv")
    parser.add_argument("--test_csv", type=str, default="test_M1_fold0.csv")
    parser.add_argument("--root_dir", type=str, default="nifti_DSA_500_NonFilter_20")

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    log_dir = Path("log")
    log_dir.mkdir(parents=True, exist_ok=True)

    log_filepath = log_dir / f"{Path(__file__).stem}.log"

    logging.basicConfig(
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
        format="%(asctime)s %(levelname)-8s %(message)s",
        handlers=[
            logging.FileHandler(log_filepath, mode="w"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logging.info(f"Using device {device}")
    print(f"Number of GPUs available: {torch.cuda.device_count()}")

    net = XLMC(
        num_classes=4,
        task="classification",
        input_type=args.input_type,
        fusion_mode="cross_attention",
        bidirectional=False,
        gru_hidden=256,
    )

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs.")
        net = nn.DataParallel(net)
        net.to("cuda:0")
    else:
        net.to(device)

    if isinstance(net, nn.DataParallel):
        logging.info(f"Model is using DataParallel across GPUs: {net.device_ids}")

    transform = A.Compose(
        [
            A.HorizontalFlip(p=0.5),

            A.ShiftScaleRotate(
                shift_limit=0.05,
                scale_limit=0.10,
                rotate_limit=10,
                border_mode=cv2.BORDER_REPLICATE,
                p=0.5,
            ),

            A.RandomBrightnessContrast(
                brightness_limit=0,
                contrast_limit=0.05,
                p=0.5,
                brightness_by_max=False,
            ),
        ],
        additional_targets={"image2": "image"},
    )

    train_set = BiplaneDSADataset(
        csv_path=args.train_csv,
        root_dir=args.root_dir,
        scale=args.img_scale,
        transform=transform,
        input_type=args.input_type,
    )

    val_set = BiplaneDSADataset(
        csv_path=args.val_csv,
        root_dir=args.root_dir,
        scale=args.img_scale,
        transform=None,
        input_type=args.input_type,
    )

    loader_args = {
        "num_workers": 0,
        "pin_memory": True,
    }

    train_loader = DataLoader(
        train_set,
        shuffle=True,
        batch_size=args.batch_size,
        **loader_args,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=1,
        shuffle=False,
        drop_last=False,
    )

    experiment = wandb.init(
        mode="online",
        project="CollateralScoring",
        resume="never",
        anonymous="must",
        group=args.wandb_project,
    )

    experiment.config.update(vars(args))
    experiment.define_metric("val_loss", summary="min")
    experiment.define_metric("train_loss", summary="min")

    if not args.output_checkpoint:
        args.output_checkpoint = f"{experiment.name}_classification_model.pt"

    if not Path(args.output_checkpoint).exists():
        train_net(
            net=net,
            epochs=args.epochs,
            amp=args.amp,
            device=device,
            wandb_logging=experiment,
            output_checkpoint=args.output_checkpoint,
        )

    checkpoint = torch.load(args.output_checkpoint, map_location=device)

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model = net.module if isinstance(net, nn.DataParallel) else net
    model_state_dict = model.state_dict()

    filtered_state_dict = {}

    for key, value in state_dict.items():
        if key in model_state_dict and value.shape == model_state_dict[key].shape:
            filtered_state_dict[key] = value
        else:
            print(f"Skipping {key}: shape mismatch or not found in model")

    model.load_state_dict(filtered_state_dict, strict=False)
    net.to(device)

    logging.info(f"Loaded checkpoint from {args.output_checkpoint}")

    val_loss, val_accuracy, val_f1, val_qwk = evaluate(
        net,
        val_loader,
        task="classification",
        device=device,
        mode="val",
        wb=wandb,
        save_predictions=True,
        output_file=f"val_predictions_{experiment.name}_classification.csv",
        focal=False,
    )

    logging.info(
        f"Validation Results -- Loss: {val_loss:.4f}, "
        f"Accuracy: {val_accuracy:.4f}, "
        f"F1: {val_f1:.4f}, "
        f"QWK: {val_qwk:.4f}"
    )

    test_set = BiplaneDSADataset(
        csv_path=args.test_csv,
        root_dir=args.root_dir,
        scale=args.img_scale,
        transform=None,
        input_type=args.input_type,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=1,
        shuffle=False,
        drop_last=False,
    )

    test_loss, test_accuracy, test_f1, test_qwk = evaluate(
        net,
        test_loader,
        task="classification",
        device=device,
        mode="test",
        wb=wandb,
        save_predictions=True,
        output_file=f"test_predictions_{experiment.name}_classification.csv",
        focal=False,
    )

    logging.info(
        f"Test Results -- Loss: {test_loss:.4f}, "
        f"Accuracy: {test_accuracy:.4f}, "
        f"F1: {test_f1:.4f}, "
        f"QWK: {test_qwk:.4f}"
    )

    results = {
        "wandb": experiment.name,
        "val_loss": val_loss,
        "val_accuracy": val_accuracy,
        "val_f1": val_f1,
        "val_qwk": val_qwk,
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "test_f1": test_f1,
        "test_qwk": test_qwk,
    }

    df_result = pd.DataFrame.from_records([results])

    df_result.to_csv(
        f"results_{experiment.name}_classification.csv",
        index=False,
        float_format="%.4f",
    )

    logging.info("Results saved to CSV. Done!")