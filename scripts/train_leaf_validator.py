#!/usr/bin/env python3
"""Leaf validator training script.

Implements a two‑stage training of a MobileNetV3‑Small binary classifier with:
- Frozen backbone for the first few epochs (head only training)
- Fine‑tuning of the backbone afterwards
- Early stopping based on validation F1
- Threshold selection using F1, balanced accuracy, and (FAR+FRR) tie‑breaker
- Comprehensive checkpoint metadata and JSON reports

All safety constraints from the user are respected: no modifications to
`model/best_model.pth`, no dataset changes, no integration into the server, and
no external package installations.
"""

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Tuple, Dict, Any, List

from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from torchvision.models import MobileNet_V3_Small_Weights

# -----------------------
# Configuration constants
# -----------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "validator_dataset"
MODEL_DIR = PROJECT_ROOT / "model"
CHECKPOINT_PATH = MODEL_DIR / "leaf_validator.pth"
REPORT_PATH = MODEL_DIR / "validator_training_report.json"
HISTORY_PATH = MODEL_DIR / "validator_training_history.json"
CMAP_PATH = MODEL_DIR / "validator_confusion_matrix.png"

SEED = 42
MAX_EPOCHS = 30
BATCH_SIZE = 64
EARLY_STOPPING_PATIENCE = 4
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Two‑stage schedule
HEAD_EPOCHS = 5          # epochs with frozen backbone (head only)
HEAD_LR = 1e-3           # learning rate for classifier head
BACKBONE_LR = 1e-4       # learning rate for backbone during fine‑tuning

# -----------------------
# Helper functions
# -----------------------
def set_seed(seed: int = SEED) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    """Return (train_transform, eval_transform)."""
    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return train_tf, eval_tf


def build_dataloaders() -> Tuple[DataLoader, DataLoader, DataLoader, Dict[int, str]]:
    """Create DataLoaders for train/val/test and return explicit label mapping.

    The ImageFolder ordering is *not* trusted; we extract the index of the
    ``leaf`` folder and build a target_transform that maps:
        NON_LEAF -> 0
        LEAF     -> 1
    The function also returns ``idx_to_label`` for later verification.
    """
    train_tf, eval_tf = get_transforms()

    # Train set – capture class_to_idx from the training split
    train_set = ImageFolder(root=str(DATASET_ROOT / "train"), transform=train_tf)
    leaf_idx = train_set.class_to_idx.get("leaf")
    non_leaf_idx = train_set.class_to_idx.get("non_leaf")
    if leaf_idx is None or non_leaf_idx is None:
        raise RuntimeError("Expected subfolders 'leaf' and 'non_leaf' in training data.")

    idx_to_label = {leaf_idx: "leaf", non_leaf_idx: "non_leaf"}
    target_transform = lambda x: 1 if x == leaf_idx else 0
    train_set.target_transform = target_transform
    train_loader = DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True
    )

    # Validation and test sets – reuse the same target_transform
    val_set = ImageFolder(root=str(DATASET_ROOT / "val"), transform=eval_tf)
    val_set.target_transform = target_transform
    val_loader = DataLoader(
        val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True
    )

    test_set = ImageFolder(root=str(DATASET_ROOT / "test"), transform=eval_tf)
    test_set.target_transform = target_transform
    test_loader = DataLoader(
        test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True
    )

    return train_loader, val_loader, test_loader, idx_to_label


def verify_class_mappings(train_loader: DataLoader, val_loader: DataLoader, test_loader: DataLoader) -> None:
    """Ensure that ``class_to_idx`` dictionaries are identical across splits.

    The function aborts with a clear error if any mismatch is found.
    """
    train_map = train_loader.dataset.class_to_idx
    val_map = val_loader.dataset.class_to_idx
    test_map = test_loader.dataset.class_to_idx
    if train_map != val_map or train_map != test_map:
        raise RuntimeError(
            f"class_to_idx mismatch between splits:\ntrain={train_map}\nval={val_map}\ntest={test_map}"
        )
    # Verify binary targets are only 0 or 1
    for loader, name in [(train_loader, "train"), (val_loader, "val"), (test_loader, "test")]:
        for _, targets in loader:
            if not torch.isin(targets, torch.tensor([0, 1])).all():
                raise RuntimeError(f"{name} loader contains targets outside {{0,1}}")
            break


def build_model() -> nn.Module:
    """MobileNetV3‑Small with pretrained ImageNet weights and a binary head."""
    try:
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
    except Exception as e:
        raise RuntimeError(
            "Pretrained MobileNetV3‑Small weights not available in this torchvision build."
        ) from e
    model = models.mobilenet_v3_small(weights=weights)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, 2)
    return model


def evaluate(
    loader: DataLoader,
    model: nn.Module,
    device: torch.device,
    threshold: float = 0.5,
) -> Tuple[int, int, int, int, List[float]]:
    """Run inference and return confusion counts.

    Returns ``(TN, FP, FN, TP, probs_list)`` where ``probs_list`` holds the
    leaf‑class probability for each sample (useful for threshold search).
    """
    model.eval()
    tn = fp = fn = tp = 0
    probs: List[float] = []
    with torch.no_grad():
        for imgs, targets in loader:
            imgs = imgs.to(device)
            targets = targets.to(device)
            logits = model(imgs)
            leaf_prob = F.softmax(logits, dim=1)[:, 1]
            preds = (leaf_prob >= threshold).long()
            tn += ((preds == 0) & (targets == 0)).sum().item()
            fp += ((preds == 1) & (targets == 0)).sum().item()
            fn += ((preds == 0) & (targets == 1)).sum().item()
            tp += ((preds == 1) & (targets == 1)).sum().item()
            probs.extend(leaf_prob.cpu().tolist())
    return tn, fp, fn, tp, probs


def compute_metrics(
    tn: int, fp: int, fn: int, tp: int
) -> Dict[str, float]:
    """Calculate all required classification metrics from the confusion matrix.
    """
    total = tn + fp + fn + tp
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0
    tpr = recall
    tnr = tn / (tn + fp) if (tn + fp) else 0.0
    balanced_accuracy = (tpr + tnr) / 2
    far = fp / (fp + tn) if (fp + tn) else 0.0
    frr = fn / (fn + tp) if (fn + tp) else 0.0
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "balanced_accuracy": balanced_accuracy,
        "far": far,
        "frr": frr,
        "sample_count": total,
    }


def select_threshold(
    loader: DataLoader,
    model: nn.Module,
    device: torch.device,
) -> Tuple[float, Dict[str, float]]:
    """Search thresholds to maximise F1 on validation data.

    Returns the best threshold together with the metrics for that threshold.
    Tie‑breaking order: higher F1 → higher balanced accuracy → lower (FAR+FRR).
    """
    model.eval()
    probs: List[float] = []
    true_labels: List[int] = []
    with torch.no_grad():
        for imgs, targets in loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            leaf_prob = F.softmax(logits, dim=1)[:, 1]
            probs.extend(leaf_prob.cpu().tolist())
            true_labels.extend(targets.tolist())

    best_thr = 0.5
    best_f1 = -1.0
    best_bal = -1.0
    best_far_frr = float("inf")
    best_metrics: Dict[str, float] = {}

    for thr in [i / 100 for i in range(5, 96)]:  # 0.05 … 0.95
        preds = [1 if p >= thr else 0 for p in probs]
        tp = sum(1 for p, t in zip(preds, true_labels) if p == 1 and t == 1)
        tn = sum(1 for p, t in zip(preds, true_labels) if p == 0 and t == 0)
        fp = sum(1 for p, t in zip(preds, true_labels) if p == 1 and t == 0)
        fn = sum(1 for p, t in zip(preds, true_labels) if p == 0 and t == 1)
        metrics = compute_metrics(tn, fp, fn, tp)
        f1 = metrics["f1"]
        bal = metrics["balanced_accuracy"]
        far_frr = metrics["far"] + metrics["frr"]
        if (
            f1 > best_f1
            or (abs(f1 - best_f1) < 1e-9 and bal > best_bal)
            or (
                abs(f1 - best_f1) < 1e-9
                and abs(bal - best_bal) < 1e-9
                and far_frr < best_far_frr
            )
        ):
            best_f1 = f1
            best_bal = bal
            best_far_frr = far_frr
            best_thr = thr
            best_metrics = metrics
    return best_thr, best_metrics


def save_report(
    config: Dict[str, Any],
    history: Dict[str, List[float]],
    threshold: float,
    val_cm: Tuple[int, int, int, int],
    test_cm: Tuple[int, int, int, int],
    test_metrics: Dict[str, float],
    val_metrics: Dict[str, float],
) -> None:
    """Write a comprehensive JSON report.
    """
    report = {
        "config": config,
        "training_history": history,
        "selected_threshold": threshold,
        "validation_confusion_matrix": [[val_cm[0], val_cm[1]], [val_cm[2], val_cm[3]]],
        "test_confusion_matrix": [[test_cm[0], test_cm[1]], [test_cm[2], test_cm[3]]],
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)


def save_history(history: Dict[str, List[float]]) -> None:
    """Persist per‑epoch training/validation loss history to JSON."""
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(HISTORY_PATH, "w") as f:
        json.dump(history, f, indent=2)


def main() -> None:
    set_seed()
    train_loader, val_loader, test_loader, idx_to_label = build_dataloaders()
    verify_class_mappings(train_loader, val_loader, test_loader)

    model = build_model().to(DEVICE)
    criterion = nn.CrossEntropyLoss()

    best_val_f1 = -1.0
    epochs_no_improve = 0
    history = {"epoch": [], "train_loss": [], "val_loss": []}

    for epoch in range(1, MAX_EPOCHS + 1):
        # Stage handling
        if epoch == 1:
            for name, param in model.named_parameters():
                if "classifier" not in name:
                    param.requires_grad = False
            optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()), lr=HEAD_LR
            )
        if epoch == HEAD_EPOCHS + 1:
            for param in model.parameters():
                param.requires_grad = True
            optimizer = torch.optim.AdamW(
                [
                    {"params": model.classifier.parameters(), "lr": HEAD_LR},
                    {"params": [p for n, p in model.named_parameters() if "classifier" not in n], "lr": BACKBONE_LR},
                ]
            )

        # Training
        model.train()
        running_loss = 0.0
        for imgs, targets in train_loader:
            imgs = imgs.to(DEVICE)
            targets = targets.to(DEVICE)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)

        # Validation loss
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, targets in val_loader:
                imgs = imgs.to(DEVICE)
                targets = targets.to(DEVICE)
                logits = model(imgs)
                loss = criterion(logits, targets)
                val_loss += loss.item() * imgs.size(0)
        val_loss /= len(val_loader.dataset)

        # Threshold selection & validation metrics
        threshold, thr_metrics = select_threshold(val_loader, model, DEVICE)
        tn, fp, fn, tp, _ = evaluate(val_loader, model, DEVICE, threshold)
        val_metrics = compute_metrics(tn, fp, fn, tp)

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            epochs_no_improve = 0
            tmp_path = CHECKPOINT_PATH.with_suffix('.tmp')
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "threshold": threshold,
                    "threshold_metrics": thr_metrics,
                    "epoch": epoch,
                    "config": {
                        "seed": SEED,
                        "max_epochs": MAX_EPOCHS,
                        "batch_size": BATCH_SIZE,
                        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
                        "device": str(DEVICE),
                        "head_epochs": HEAD_EPOCHS,
                        "head_lr": HEAD_LR,
                        "backbone_lr": BACKBONE_LR,
                    },
                },
                tmp_path,
            )
            tmp_path.replace(CHECKPOINT_PATH)
        else:
            epochs_no_improve += 1

        # Record history
        history["epoch"].append(epoch)
        history["train_loss"].append(epoch_loss)
        history["val_loss"].append(val_loss)

        print(
            f"Epoch {epoch:02d}: train_loss={epoch_loss:.4f} val_loss={val_loss:.4f} "
            f"val_f1={val_metrics['f1']:.4f} thr={threshold:.2f}"
        )

        if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
            print("Early stopping triggered.")
            break

    # Final evaluation
    checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu')
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    final_threshold = checkpoint["threshold"]

    dummy = torch.randn(1, 3, 224, 224).to(DEVICE)
    with torch.no_grad():
        out = model(dummy)
    assert out.shape[-1] == 2, "Model does not produce 2‑class logits"

    val_cm = evaluate(val_loader, model, DEVICE, final_threshold)[:4]
    test_cm = evaluate(test_loader, model, DEVICE, final_threshold)[:4]

    # compute metrics for both splits
    val_metrics = compute_metrics(*val_cm)
    test_metrics = compute_metrics(*test_cm)

    config = {
        "seed": SEED,
        "max_epochs": MAX_EPOCHS,
        "batch_size": BATCH_SIZE,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "device": str(DEVICE),
        "head_epochs": HEAD_EPOCHS,
        "head_lr": HEAD_LR,
        "backbone_lr": BACKBONE_LR,
    }
    save_report(config, history, final_threshold, val_cm, test_cm, test_metrics, val_metrics)
    save_history(history)

    # Optional confusion‑matrix plot
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for ax, cm, title in zip(
            axes,
            [val_cm, test_cm],
            ["Validation", "Test"],
        ):
            matrix = [[cm[0], cm[1]], [cm[2], cm[3]]]
            sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            ax.set_title(f"{title} Confusion Matrix")
            ax.set_xticklabels(["NON_LEAF", "LEAF"], rotation=0)
            ax.set_yticklabels(["NON_LEAF", "LEAF"], rotation=0)
        plt.tight_layout()
        fig.savefig(CMAP_PATH)
        plt.close(fig)
    except Exception as e:
        print("Failed to generate confusion‑matrix plot:", e)

    print("Training complete. Checkpoint saved at:", CHECKPOINT_PATH)
    print("Report saved at:", REPORT_PATH)
    print("History saved at:", HISTORY_PATH)
    print("Confusion‑matrix plot saved at:", CMAP_PATH)


if __name__ == "__main__":
    main()
