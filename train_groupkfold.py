"""Grouped five-fold cross-validation for architecture-aware models.

This script evaluates one manually specified HPO configuration with grouped
five-fold cross-validation. Groups are kept intact across folds to prevent
batch-level leakage. The model definition, fine-tuning scopes, augmentation,
optimizer, scheduler, early stopping, mixed precision, metrics, threshold
scan, and output files are intentionally preserved in this public script.

Usage
-----
1. Install PyTorch, timm, Albumentations, OpenCV, scikit-learn, pandas,
   NumPy, matplotlib, and seaborn.
2. Replace every placeholder in ``BACKBONE_NAME``, ``MANUAL_CONFIG``, and
   ``BASE_DIR`` with the selected HPO configuration and grouped dataset path.
3. Organize the grouped dataset as::

       grouped_dataset_root/
       ├── group_001/
       │   ├── phase/
       │   └── uniform/
       ├── group_002/
       │   ├── phase/
       │   └── uniform/
       └── ...

4. Run ``python train_groupkfold.py``. Outputs are
   written to the current working directory.

The class order is fixed as ``phase = 0`` and ``uniform = 1``. The pooled OOF
threshold is selected by maximizing binary F1 for the positive class
``uniform`` and must be fixed before independent test evaluation.
"""

import os
import gc
import cv2
import copy
import json
import random
import platform
from importlib import metadata as importlib_metadata

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    accuracy_score,
    confusion_matrix,
    classification_report,
    precision_recall_fscore_support,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold
import matplotlib.pyplot as plt
import seaborn as sns


# =============================================================================
# 1. Randomness control
# =============================================================================
SEED = 42


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(SEED)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def create_grad_scaler():
    if not AMP_ENABLED:
        return None
    try:
        return torch.amp.GradScaler("cuda", enabled=True)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=True)


def autocast_context():
    if AMP_ENABLED:
        return torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=True,
        )
    from contextlib import nullcontext
    return nullcontext()


# =============================================================================
# 2. Manual configuration
# =============================================================================
# Replace these placeholders with values selected by the HPO stage.
BACKBONE_NAME = "<BACKBONE_NAME>"

MANUAL_CONFIG = {
    "img_size": "<IMAGE_SIZE>",
    "lr": "<LEARNING_RATE>",
    "fine_tuning_scope": "<FINE_TUNING_SCOPE>",
    "batch_size": "<BATCH_SIZE>",
    "weight_decay": "<WEIGHT_DECAY>",
}

BASE_DIR = "<GROUPED_DATASET_DIR>"

EPOCHS = 25
EARLY_STOP_PATIENCE = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Mixed precision matches the HPO stage; CPU execution falls back to FP32.
USE_MIXED_PRECISION = True
AMP_ENABLED = bool(USE_MIXED_PRECISION and torch.cuda.is_available())
PRECISION_DESCRIPTION = "16-mixed" if AMP_ENABLED else "32-true"

# Each fold uses independent generators.
NUM_WORKERS = min(4, os.cpu_count() or 2)
PREFETCH_FACTOR = 2

AUGMENTATION_CONFIG_ID = "grayscale_albumentations_v1"


def validate_public_configuration():
    values = {
        "BACKBONE_NAME": BACKBONE_NAME,
        "img_size": MANUAL_CONFIG["img_size"],
        "lr": MANUAL_CONFIG["lr"],
        "fine_tuning_scope": MANUAL_CONFIG["fine_tuning_scope"],
        "batch_size": MANUAL_CONFIG["batch_size"],
        "weight_decay": MANUAL_CONFIG["weight_decay"],
        "BASE_DIR": BASE_DIR,
    }
    unresolved = [
        name
        for name, value in values.items()
        if isinstance(value, str) and value.startswith("<")
    ]
    if unresolved:
        raise ValueError(
            "Replace the public-release placeholders before running: "
            + ", ".join(unresolved)
        )


validate_public_configuration()


# =============================================================================
# 3. timm model mapping
# =============================================================================
TIMM_MODEL_NAMES = {
    "ResNet18": "resnet18",
    "ResNet50": "resnet50",
    "VGG16": "vgg16",
    "DenseNet121": "densenet121",
    "EfficientNetB0": "efficientnet_b0",
    "MobileNetV2": "mobilenetv2_100",
    "ConvNeXt_Tiny": "convnext_tiny",
}

if BACKBONE_NAME not in TIMM_MODEL_NAMES:
    raise ValueError(
        f"Unsupported BACKBONE_NAME: {BACKBONE_NAME}. "
        f"Available models: {list(TIMM_MODEL_NAMES)}"
    )

MODEL_TO_RUN = TIMM_MODEL_NAMES[BACKBONE_NAME]


# =============================================================================
# 4. Architecture-aware fine-tuning scopes
# =============================================================================
# This list is identical to the architecture-aware HPO script.
FINE_TUNING_SCOPES = {
    "ResNet18": [
        "head_only",
        "layer4_last1",
        "layer4",
        "layer3_last1_plus_layer4",
        "layer3_plus_layer4",
        "all",
    ],
    "ResNet50": [
        "head_only",
        "layer4_last1",
        "layer4",
        "layer3_last3_plus_layer4",
        "layer3_plus_layer4",
        "all",
    ],
    "DenseNet121": [
        "head_only",
        "denseblock4_last4",
        "denseblock4_last8",
        "denseblock4_last13",
        "denseblock4",
        "transition3_plus_denseblock4",
        "all",
    ],
    "EfficientNetB0": [
        "head_only",
        "stage7",
        "stage6_to_7",
        "stage5_to_7",
        "stage4_last2_plus_stage5_to_7",
        "stage4_to_7",
        "all",
    ],
    "MobileNetV2": [
        "head_only",
        "stage7",
        "stage6_to_7",
        "stage5_to_7",
        "stage4_to_7",
        "stage3_to_7",
        "all",
    ],
    "VGG16": [
        "classifier_only",
        "block5",
        "conv4_3_plus_block5",
        "block4_plus_block5",
        "block3_to_block5",
        "all",
    ],
    "ConvNeXt_Tiny": [
        "head_only",
        "stage4",
        "stage3_last3_plus_stage4",
        "stage3_last6_plus_stage4",
        "stage3_last8_plus_stage4",
        "stage3_plus_stage4",
        "all",
    ],
}


# =============================================================================
# 5. Validate and use the manual configuration
# =============================================================================
SELECTED_CONFIG = copy.deepcopy(MANUAL_CONFIG)
SELECTED_CONFIG["configuration_source"] = "manual configuration at script top"
SELECTED_CONFIG["expected_trainable_parameter_ratio"] = None
SELECTED_CONFIG["hpo_best_validation_auc"] = None
SELECTED_CONFIG["hpo_best_validation_epoch"] = None

required_manual_keys = {
    "img_size",
    "lr",
    "fine_tuning_scope",
    "batch_size",
    "weight_decay",
}
missing_manual_keys = required_manual_keys.difference(SELECTED_CONFIG)
if missing_manual_keys:
    raise ValueError(
        f"MANUAL_CONFIG is missing required keys: {sorted(missing_manual_keys)}"
    )

if SELECTED_CONFIG["fine_tuning_scope"] not in FINE_TUNING_SCOPES[BACKBONE_NAME]:
    raise ValueError(
        f"Invalid fine_tuning_scope '{SELECTED_CONFIG['fine_tuning_scope']}' "
        f"for {BACKBONE_NAME}. Valid scopes: "
        f"{FINE_TUNING_SCOPES[BACKBONE_NAME]}"
    )

IMG_SIZE = int(SELECTED_CONFIG["img_size"])

CLASS_NAMES = ["phase", "uniform"]
CLASS_TO_IDX = {class_name: idx for idx, class_name in enumerate(CLASS_NAMES)}


# =============================================================================
# 6. Dataset parsing and preprocessing
# =============================================================================
def parse_dataset_to_df(base_path):
    data = []
    for exp_folder in os.listdir(base_path):
        exp_path = os.path.join(base_path, exp_folder)
        if not os.path.isdir(exp_path):
            continue

        for class_name in CLASS_NAMES:
            class_path = os.path.join(exp_path, class_name)
            if not os.path.isdir(class_path):
                continue

            for image_name in os.listdir(class_path):
                image_path = os.path.join(class_path, image_name)
                data.append({
                    "filepath": image_path,
                    "label": CLASS_TO_IDX[class_name],
                    "group": exp_folder,
                })

    return pd.DataFrame(data)


class CustomDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        path = self.df.loc[idx, "filepath"]
        label = self.df.loc[idx, "label"]

        # Match HPO: read one grayscale channel and replicate it to RGB.
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Failed to read image: {path}")
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        if self.transform:
            image = self.transform(image=image)["image"]

        return image, torch.tensor(label, dtype=torch.float32)


def get_transforms():
    # Use the same augmentation parameters as the two-stage HPO script.
    train_transform = A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.Affine(
            scale=(0.95, 1.05),
            translate_percent=(-0.05, 0.05),
            rotate=(-30, 30),
            border_mode=cv2.BORDER_REFLECT_101,
            p=0.5,
        ),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.5,
        ),
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
        ToTensorV2(),
    ])

    validation_transform = A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
        ToTensorV2(),
    ])

    return train_transform, validation_transform


# =============================================================================
# 7. Architecture-aware fine-tuning utilities
# =============================================================================
def set_module_trainable(module, trainable=True):
    for parameter in module.parameters():
        parameter.requires_grad = trainable


def unfreeze_modules(modules):
    for module in modules:
        set_module_trainable(module, trainable=True)


def get_parameter_statistics(model):
    total_params = sum(parameter.numel() for parameter in model.parameters())
    trainable_params = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    if total_params == 0:
        raise RuntimeError("The model contains no parameters.")

    trainable_names = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "trainable_parameter_ratio": trainable_params / total_params,
        "first_trainable_parameter": (
            trainable_names[0] if trainable_names else "NONE"
        ),
        "last_trainable_parameter": (
            trainable_names[-1] if trainable_names else "NONE"
        ),
    }


def apply_fine_tuning_scope(model, backbone_name, fine_tuning_scope):
    """
    Match the HPO implementation: freeze the full model first, then unfreeze
    complete architecture-aware units. Every scope includes the classifier.
    """
    valid_scopes = FINE_TUNING_SCOPES.get(backbone_name)
    if valid_scopes is None:
        raise ValueError(f"No scope list is defined for {backbone_name}.")
    if fine_tuning_scope not in valid_scopes:
        raise ValueError(
            f"Invalid scope '{fine_tuning_scope}' for {backbone_name}. "
            f"Valid scopes: {valid_scopes}"
        )

    set_module_trainable(model, trainable=False)

    if fine_tuning_scope == "all":
        set_module_trainable(model, trainable=True)
        return get_parameter_statistics(model)

    # ------------------------- ResNet18 / ResNet50 -------------------------
    if backbone_name.startswith("ResNet"):
        set_module_trainable(model.fc, trainable=True)

        if fine_tuning_scope == "head_only":
            pass
        elif fine_tuning_scope == "layer4_last1":
            set_module_trainable(model.layer4[-1], trainable=True)
        elif fine_tuning_scope == "layer4":
            set_module_trainable(model.layer4, trainable=True)
        elif fine_tuning_scope == "layer3_last1_plus_layer4":
            unfreeze_modules([
                model.layer3[-1],
                model.layer4,
            ])
        elif fine_tuning_scope == "layer3_last3_plus_layer4":
            unfreeze_modules([
                *list(model.layer3.children())[-3:],
                model.layer4,
            ])
        elif fine_tuning_scope == "layer3_plus_layer4":
            unfreeze_modules([
                model.layer3,
                model.layer4,
            ])

    # ------------------------------ DenseNet121 ----------------------------
    elif backbone_name == "DenseNet121":
        denseblock4_layers = list(model.features.denseblock4.children())
        if len(denseblock4_layers) != 16:
            raise RuntimeError(
                "Unexpected timm DenseNet121 denseblock4 structure: "
                f"expected 16 dense layers, got {len(denseblock4_layers)}."
            )

        unfreeze_modules([
            model.features.norm5,
            model.classifier,
        ])

        if fine_tuning_scope == "head_only":
            pass
        elif fine_tuning_scope == "denseblock4_last4":
            unfreeze_modules(denseblock4_layers[-4:])
        elif fine_tuning_scope == "denseblock4_last8":
            unfreeze_modules(denseblock4_layers[-8:])
        elif fine_tuning_scope == "denseblock4_last13":
            unfreeze_modules(denseblock4_layers[-13:])
        elif fine_tuning_scope == "denseblock4":
            set_module_trainable(model.features.denseblock4, trainable=True)
        elif fine_tuning_scope == "transition3_plus_denseblock4":
            unfreeze_modules([
                model.features.transition3,
                model.features.denseblock4,
            ])

    # ----------------------------- EfficientNetB0 --------------------------
    elif backbone_name == "EfficientNetB0":
        if len(model.blocks) != 7:
            raise RuntimeError(
                "Unexpected timm EfficientNetB0 block structure: "
                f"expected 7 stages, got {len(model.blocks)}."
            )

        unfreeze_modules([
            model.conv_head,
            model.bn2,
            model.classifier,
        ])

        if fine_tuning_scope == "head_only":
            pass
        elif fine_tuning_scope == "stage7":
            set_module_trainable(model.blocks[6], trainable=True)
        elif fine_tuning_scope == "stage6_to_7":
            unfreeze_modules(model.blocks[5:7])
        elif fine_tuning_scope == "stage5_to_7":
            unfreeze_modules(model.blocks[4:7])
        elif fine_tuning_scope == "stage4_last2_plus_stage5_to_7":
            stage4_blocks = list(model.blocks[3].children())
            if len(stage4_blocks) < 2:
                raise RuntimeError(
                    "EfficientNetB0 stage 4 contains fewer than two blocks."
                )
            unfreeze_modules(stage4_blocks[-2:])
            unfreeze_modules(model.blocks[4:7])
        elif fine_tuning_scope == "stage4_to_7":
            unfreeze_modules(model.blocks[3:7])

    # ------------------------------ MobileNetV2 ----------------------------
    elif backbone_name == "MobileNetV2":
        if len(model.blocks) != 7:
            raise RuntimeError(
                "Unexpected timm MobileNetV2 block structure: "
                f"expected 7 stages, got {len(model.blocks)}."
            )

        unfreeze_modules([
            model.conv_head,
            model.bn2,
            model.classifier,
        ])

        if fine_tuning_scope == "head_only":
            pass
        elif fine_tuning_scope == "stage7":
            set_module_trainable(model.blocks[6], trainable=True)
        elif fine_tuning_scope == "stage6_to_7":
            unfreeze_modules(model.blocks[5:7])
        elif fine_tuning_scope == "stage5_to_7":
            unfreeze_modules(model.blocks[4:7])
        elif fine_tuning_scope == "stage4_to_7":
            unfreeze_modules(model.blocks[3:7])
        elif fine_tuning_scope == "stage3_to_7":
            unfreeze_modules(model.blocks[2:7])

    # -------------------------------- VGG16 --------------------------------
    elif backbone_name == "VGG16":
        if len(model.features) != 31:
            raise RuntimeError(
                "Unexpected timm VGG16 feature structure: "
                f"expected 31 modules, got {len(model.features)}."
            )

        unfreeze_modules([
            model.pre_logits,
            model.head,
        ])

        if fine_tuning_scope == "classifier_only":
            pass
        elif fine_tuning_scope == "block5":
            unfreeze_modules(model.features[24:31])
        elif fine_tuning_scope == "conv4_3_plus_block5":
            unfreeze_modules(model.features[21:31])
        elif fine_tuning_scope == "block4_plus_block5":
            unfreeze_modules(model.features[17:31])
        elif fine_tuning_scope == "block3_to_block5":
            unfreeze_modules(model.features[10:31])

    # ----------------------------- ConvNeXt-Tiny ---------------------------
    elif backbone_name == "ConvNeXt_Tiny":
        if len(model.stages) != 4:
            raise RuntimeError(
                "Unexpected timm ConvNeXt-Tiny stage structure: "
                f"expected 4 stages, got {len(model.stages)}."
            )

        stage3_blocks = list(model.stages[2].blocks.children())
        if len(stage3_blocks) != 9:
            raise RuntimeError(
                "Unexpected timm ConvNeXt-Tiny stage 3 structure: "
                f"expected 9 blocks, got {len(stage3_blocks)}."
            )

        set_module_trainable(model.head, trainable=True)

        if fine_tuning_scope == "head_only":
            pass
        elif fine_tuning_scope == "stage4":
            set_module_trainable(model.stages[3], trainable=True)
        elif fine_tuning_scope == "stage3_last3_plus_stage4":
            unfreeze_modules([
                *stage3_blocks[-3:],
                model.stages[3],
            ])
        elif fine_tuning_scope == "stage3_last6_plus_stage4":
            unfreeze_modules([
                *stage3_blocks[-6:],
                model.stages[3],
            ])
        elif fine_tuning_scope == "stage3_last8_plus_stage4":
            unfreeze_modules([
                *stage3_blocks[-8:],
                model.stages[3],
            ])
        elif fine_tuning_scope == "stage3_plus_stage4":
            unfreeze_modules([
                model.stages[2],
                model.stages[3],
            ])

    else:
        raise ValueError(f"Unsupported backbone: {backbone_name}")

    statistics = get_parameter_statistics(model)
    if statistics["trainable_params"] == 0:
        raise RuntimeError(
            f"Scope '{fine_tuning_scope}' left no trainable parameters."
        )

    return statistics


def set_frozen_batchnorm_eval(model):
    """
    Match HPO behavior by keeping the running statistics of frozen BatchNorm
    layers in evaluation mode after ``model.train()`` is called.
    """
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            affine_parameters = list(module.parameters(recurse=False))
            if affine_parameters and not any(
                parameter.requires_grad
                for parameter in affine_parameters
            ):
                module.eval()


# =============================================================================
# 8. Core per-fold outputs
# =============================================================================
def save_comprehensive_outputs(history_dict, y_true, y_prob, prefix, class_names):
    # Save learning-curve data.
    df_history = pd.DataFrame(history_dict)
    df_history.to_csv(f"{prefix}_learning_curve.csv", index=False)

    # Save ROC-curve data.
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    pd.DataFrame({
        "FPR": fpr,
        "TPR": tpr,
        "Threshold": roc_thresholds,
    }).to_csv(f"{prefix}_roc_curve_data.csv", index=False)

    # Save precision-recall curve data.
    prec, rec, pr_thresholds = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(rec, prec)
    temp_thresholds = np.append(pr_thresholds, 1.0)
    pd.DataFrame({
        "Precision": prec,
        "Recall": rec,
        "Threshold": temp_thresholds,
    }).to_csv(f"{prefix}_pr_curve_data.csv", index=False)

    # Save the evaluation report.
    best_t = 0.5
    y_pred = (y_prob > best_t).astype(int)
    acc = accuracy_score(y_true, y_pred)
    p_val, r_val, f1_val, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        zero_division=0,
    )

    with open(f"{prefix}_evaluation_report.txt", "w") as file:
        file.write(f"=== Model Evaluation Report: {prefix} ===\n")
        file.write(f"ROC AUC: {roc_auc:.4f}\n")
        file.write(f"PR AUC: {pr_auc:.4f}\n")
        file.write(f"Accuracy (T=0.5): {acc:.4f}\n")
        file.write(
            "Binary Precision: "
            f"{p_val:.4f}\n"
        )
        file.write(
            "Binary Recall: "
            f"{r_val:.4f}\n"
        )
        file.write(
            "Binary F1: "
            f"{f1_val:.4f}\n\n"
        )
        file.write("Classification Report:\n")
        file.write(
            classification_report(
                y_true,
                y_pred,
                target_names=class_names,
            )
        )

    # Save the confusion matrix.
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title(f"{prefix} - Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(f"{prefix}_confusion_matrix.png", dpi=150)
    plt.close()

    # Save learning, ROC, and precision-recall curves.
    fig, axes = plt.subplots(1, 3, figsize=(21, 6))
    if not df_history.empty:
        axes[0].plot(
            df_history["accuracy"],
            label="Train Acc",
            color="tab:blue",
            lw=2,
        )
        axes[0].plot(
            df_history["val_accuracy"],
            label="Val Acc",
            color="tab:orange",
            lw=2,
        )
        axes[0].set_ylabel("Accuracy", color="tab:blue")
        ax2 = axes[0].twinx()
        ax2.plot(
            df_history["loss"],
            label="Train Loss",
            color="tab:green",
            lw=1.5,
        )
        ax2.plot(
            df_history["val_loss"],
            label="Val Loss",
            color="tab:red",
            lw=1.5,
        )
        ax2.set_ylabel("Loss", color="tab:red")
        axes[0].set_title("Learning Curve")
        axes[0].legend(loc="upper left")
        ax2.legend(loc="upper right")

    axes[1].plot(
        fpr,
        tpr,
        color="darkorange",
        lw=2,
        label=f"ROC (AUC = {roc_auc:.4f})",
    )
    axes[1].plot([0, 1], [0, 1], linestyle="--")
    axes[1].set_title("ROC Curve")
    axes[1].set_xlabel("FPR")
    axes[1].set_ylabel("TPR")
    axes[1].legend()

    axes[2].plot(
        rec,
        prec,
        color="green",
        lw=2,
        label=f"PR (AUC = {pr_auc:.4f})",
    )
    axes[2].set_title("PR Curve")
    axes[2].set_xlabel("Recall")
    axes[2].set_ylabel("Precision")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(f"{prefix}_comprehensive_curves.png", dpi=150)
    plt.close()

    return roc_auc, acc, f1_val


# =============================================================================
# 9. Additional audit and pooled-OOF outputs
# =============================================================================
def safe_package_version(distribution_name):
    try:
        return importlib_metadata.version(distribution_name)
    except importlib_metadata.PackageNotFoundError:
        return "not installed / unavailable"


def save_cv_configuration(parameter_statistics):
    gpu_name = (
        torch.cuda.get_device_name(0)
        if torch.cuda.is_available()
        else None
    )

    report = {
        "pipeline_stage": "grouped five-fold cross-validation",
        "backbone": BACKBONE_NAME,
        "timm_model_name": MODEL_TO_RUN,
        "model_implementation": "timm",
        "configuration_source": SELECTED_CONFIG["configuration_source"],
        "fine_tuning_scope": SELECTED_CONFIG["fine_tuning_scope"],
        "image_size": IMG_SIZE,
        "learning_rate": SELECTED_CONFIG["lr"],
        "weight_decay": SELECTED_CONFIG["weight_decay"],
        "batch_size": SELECTED_CONFIG["batch_size"],
        "epochs": EPOCHS,
        "early_stop_patience": EARLY_STOP_PATIENCE,
        "early_stopping_monitor": "validation ROC-AUC",
        "best_checkpoint_monitor": "validation ROC-AUC",
        "total_params": parameter_statistics["total_params"],
        "trainable_params": parameter_statistics["trainable_params"],
        "trainable_parameter_ratio": parameter_statistics[
            "trainable_parameter_ratio"
        ],
        "expected_hpo_trainable_parameter_ratio": SELECTED_CONFIG[
            "expected_trainable_parameter_ratio"
        ],
        "first_trainable_parameter": parameter_statistics[
            "first_trainable_parameter"
        ],
        "last_trainable_parameter": parameter_statistics[
            "last_trainable_parameter"
        ],
        "classes": CLASS_NAMES,
        "class_to_idx": CLASS_TO_IDX,
        "positive_class": "uniform",
        "base_seed": SEED,
        "fold_seed_rule": "SEED + fold - 1",
        "precision": PRECISION_DESCRIPTION,
        "augmentation_config_id": AUGMENTATION_CONFIG_ID,
        "software_versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "timm": getattr(
                timm,
                "__version__",
                safe_package_version("timm"),
            ),
            "albumentations": getattr(
                A,
                "__version__",
                safe_package_version("albumentations"),
            ),
            "opencv": cv2.__version__,
            "scikit-learn": safe_package_version("scikit-learn"),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "matplotlib": safe_package_version("matplotlib"),
            "seaborn": safe_package_version("seaborn"),
        },
        "hardware": {
            "device": str(DEVICE),
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "gpu_name": gpu_name,
        },
    }

    with open(
        f"CV_CONFIGURATION_{MODEL_TO_RUN}.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(report, file, indent=2, ensure_ascii=False)


def save_fold_additional_outputs(
    fold,
    prefix,
    train_df,
    val_df,
    y_true,
    y_prob,
    extended_history,
    parameter_statistics,
    best_epoch,
    best_auc,
    best_checkpoint_val_loss,
    fold_seed,
):
    y_pred = (y_prob > 0.5).astype(int)

    prediction_df = val_df.reset_index(drop=True).copy()
    prediction_df["fold"] = fold
    prediction_df["true_class"] = [
        CLASS_NAMES[int(value)] for value in y_true
    ]
    prediction_df["probability_uniform"] = y_prob
    prediction_df["predicted_label_T0.5"] = y_pred
    prediction_df["predicted_class_T0.5"] = [
        CLASS_NAMES[int(value)] for value in y_pred
    ]
    prediction_df.to_csv(
        f"{prefix}_predictions.csv",
        index=False,
    )

    pd.DataFrame(extended_history).to_csv(
        f"{prefix}_epoch_metrics_extended.csv",
        index=False,
    )

    cm = confusion_matrix(y_true, y_pred)
    pd.DataFrame(
        cm,
        index=[f"true_{name}" for name in CLASS_NAMES],
        columns=[f"pred_{name}" for name in CLASS_NAMES],
    ).to_csv(f"{prefix}_confusion_matrix_data.csv")

    train_class_counts = train_df["label"].value_counts().to_dict()
    val_class_counts = val_df["label"].value_counts().to_dict()

    with open(
        f"{prefix}_training_metadata.txt",
        "w",
        encoding="utf-8",
    ) as file:
        file.write(f"Fold: {fold}\n")
        file.write(f"Backbone: {BACKBONE_NAME}\n")
        file.write(f"timm model name: {MODEL_TO_RUN}\n")
        file.write(
            "Configuration source: "
            f"{SELECTED_CONFIG['configuration_source']}\n"
        )
        file.write(
            "Fine-tuning scope: "
            f"{SELECTED_CONFIG['fine_tuning_scope']}\n"
        )
        file.write(
            "Trainable parameter ratio: "
            f"{parameter_statistics['trainable_parameter_ratio']:.10f}\n"
        )
        file.write(
            f"Trainable parameters: "
            f"{parameter_statistics['trainable_params']:,}\n"
        )
        file.write(
            f"Total parameters: "
            f"{parameter_statistics['total_params']:,}\n"
        )
        file.write(
            "First trainable parameter: "
            f"{parameter_statistics['first_trainable_parameter']}\n"
        )
        file.write(
            "Last trainable parameter: "
            f"{parameter_statistics['last_trainable_parameter']}\n"
        )
        file.write(f"Train samples: {len(train_df)}\n")
        file.write(f"Validation samples: {len(val_df)}\n")
        file.write(
            f"Train groups: {train_df['group'].nunique()}\n"
        )
        file.write(
            f"Validation groups: {val_df['group'].nunique()}\n"
        )
        file.write(f"Train class counts: {train_class_counts}\n")
        file.write(f"Validation class counts: {val_class_counts}\n")
        file.write(f"Best validation-AUC epoch: {best_epoch}\n")
        file.write(f"Best validation ROC-AUC: {best_auc:.10f}\n")
        file.write(
            "Validation loss at best-AUC checkpoint: "
            f"{best_checkpoint_val_loss:.10f}\n"
        )
        file.write(f"Fold seed: {fold_seed}\n")
        file.write(f"Precision: {PRECISION_DESCRIPTION}\n")
        file.write(f"Augmentation config: {AUGMENTATION_CONFIG_ID}\n")
        file.write(f"Epochs completed: {len(extended_history)}\n")

    return prediction_df


def save_oof_additional_outputs(
    all_oof_y_true,
    all_oof_y_prob,
    all_oof_records,
    threshold_scan_df,
    best_threshold,
):
    all_oof_y_true = np.asarray(all_oof_y_true)
    all_oof_y_prob = np.asarray(all_oof_y_prob)
    final_pred = (all_oof_y_prob > best_threshold).astype(int)

    oof_df = pd.concat(all_oof_records, ignore_index=True)
    oof_df["predicted_label_best_threshold"] = final_pred
    oof_df["predicted_class_best_threshold"] = [
        CLASS_NAMES[int(value)] for value in final_pred
    ]
    oof_df["best_threshold"] = best_threshold
    oof_df.to_csv(
        f"CV_OOF_PREDICTIONS_{MODEL_TO_RUN}.csv",
        index=False,
    )

    threshold_scan_df.to_csv(
        f"CV_THRESHOLD_SCAN_{MODEL_TO_RUN}.csv",
        index=False,
    )

    fpr, tpr, roc_thresholds = roc_curve(
        all_oof_y_true,
        all_oof_y_prob,
    )
    roc_auc_value = auc(fpr, tpr)
    pd.DataFrame({
        "FPR": fpr,
        "TPR": tpr,
        "Threshold": roc_thresholds,
    }).to_csv(
        f"CV_OOF_ROC_CURVE_DATA_{MODEL_TO_RUN}.csv",
        index=False,
    )

    precision_values, recall_values, pr_thresholds = precision_recall_curve(
        all_oof_y_true,
        all_oof_y_prob,
    )
    pr_auc_value = auc(recall_values, precision_values)
    pd.DataFrame({
        "Precision": precision_values,
        "Recall": recall_values,
        "Threshold": np.append(pr_thresholds, 1.0),
    }).to_csv(
        f"CV_OOF_PR_CURVE_DATA_{MODEL_TO_RUN}.csv",
        index=False,
    )

    final_acc = accuracy_score(all_oof_y_true, final_pred)
    final_binary_precision, final_binary_recall, final_binary_f1, _ = (
        precision_recall_fscore_support(
            all_oof_y_true,
            final_pred,
            average="binary",
            zero_division=0,
        )
    )
    final_macro_f1 = f1_score(
        all_oof_y_true,
        final_pred,
        average="macro",
        zero_division=0,
    )

    with open(
        f"CV_OOF_EVALUATION_REPORT_{MODEL_TO_RUN}.txt",
        "w",
        encoding="utf-8",
    ) as file:
        file.write(
            f"=== Pooled OOF Evaluation: {MODEL_TO_RUN} ===\n"
        )
        file.write(f"Best OOF threshold: {best_threshold:.4f}\n")
        file.write(f"ROC AUC: {roc_auc_value:.4f}\n")
        file.write(f"PR AUC: {pr_auc_value:.4f}\n")
        file.write(f"Accuracy: {final_acc:.4f}\n")
        file.write(
            "Binary Precision: "
            f"{final_binary_precision:.4f}\n"
        )
        file.write(
            "Binary Recall: "
            f"{final_binary_recall:.4f}\n"
        )
        file.write(
            "Binary F1: "
            f"{final_binary_f1:.4f}\n"
        )
        file.write(f"Macro-F1: {final_macro_f1:.4f}\n\n")
        file.write("Classification Report:\n")
        file.write(
            classification_report(
                all_oof_y_true,
                final_pred,
                target_names=CLASS_NAMES,
            )
        )

    cm = confusion_matrix(all_oof_y_true, final_pred)
    pd.DataFrame(
        cm,
        index=[f"true_{name}" for name in CLASS_NAMES],
        columns=[f"pred_{name}" for name in CLASS_NAMES],
    ).to_csv(
        f"CV_OOF_CONFUSION_MATRIX_DATA_{MODEL_TO_RUN}.csv"
    )

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
    )
    plt.title(
        f"{MODEL_TO_RUN} - Pooled OOF Confusion Matrix\n"
        f"Threshold = {best_threshold:.2f}"
    )
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(
        f"CV_OOF_CONFUSION_MATRIX_{MODEL_TO_RUN}.png",
        dpi=150,
    )
    plt.close()

    plt.figure(figsize=(7, 6))
    plt.plot(
        fpr,
        tpr,
        lw=2,
        label=f"ROC (AUC = {roc_auc_value:.4f})",
    )
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{MODEL_TO_RUN} - Pooled OOF ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        f"CV_OOF_ROC_CURVE_{MODEL_TO_RUN}.png",
        dpi=150,
    )
    plt.close()

    plt.figure(figsize=(7, 6))
    plt.plot(
        recall_values,
        precision_values,
        lw=2,
        label=f"PR (AUC = {pr_auc_value:.4f})",
    )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"{MODEL_TO_RUN} - Pooled OOF PR Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        f"CV_OOF_PR_CURVE_{MODEL_TO_RUN}.png",
        dpi=150,
    )
    plt.close()


# =============================================================================
# 10. Cross-validation workflow
# =============================================================================
def train_and_evaluate_cv():
    train_transform, validation_transform = get_transforms()
    data_path = (
        f"{BASE_DIR}/train"
        if os.path.exists(f"{BASE_DIR}/train")
        else BASE_DIR
    )

    print("Parsing grouped dataset...")
    df_data = parse_dataset_to_df(data_path)
    print(
        f"Found {len(df_data)} images across "
        f"{df_data['group'].nunique()} independent groups."
    )

    if df_data.empty:
        raise RuntimeError("No images were found in the grouped dataset.")

    cfg = SELECTED_CONFIG

    print("\nCurrent five-fold configuration:")
    print(f"Backbone: {BACKBONE_NAME}")
    print(f"timm model: {MODEL_TO_RUN}")
    print(f"Configuration source: {cfg['configuration_source']}")
    print(f"Fine-tuning scope: {cfg['fine_tuning_scope']}")
    print(f"Image size: {IMG_SIZE}")
    print(f"Learning rate: {cfg['lr']:.6g}")
    print(f"Weight decay: {cfg['weight_decay']:.6g}")
    print(f"Batch size: {cfg['batch_size']}")
    print(f"Precision: {PRECISION_DESCRIPTION}")
    print(f"Augmentation config: {AUGMENTATION_CONFIG_ID}")

    # Validate the selected scope with a reference model before CV.
    reference_model = timm.create_model(
        MODEL_TO_RUN,
        pretrained=False,
        num_classes=1,
    )
    reference_statistics = apply_fine_tuning_scope(
        reference_model,
        BACKBONE_NAME,
        cfg["fine_tuning_scope"],
    )


    print(
        "Architecture-aware scope validation complete. "
        f"Trainable parameters: {reference_statistics['trainable_params']:,}/"
        f"{reference_statistics['total_params']:,} "
        f"({reference_statistics['trainable_parameter_ratio']:.6f})"
    )

    save_cv_configuration(reference_statistics)

    del reference_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    group_kfold = GroupKFold(n_splits=5)
    summary_log = []
    fold_detail_rows = []
    split_assignment_rows = []
    all_oof_y_true = []
    all_oof_y_prob = []
    all_oof_records = []

    for fold, (train_idx, val_idx) in enumerate(
        group_kfold.split(
            X=df_data,
            y=df_data["label"],
            groups=df_data["group"],
        ),
        start=1,
    ):
        fold_seed = SEED + fold - 1
        set_seed(fold_seed)

        print(
            f"\nStarting fold {fold}/5 "
            f"(Model: {MODEL_TO_RUN}, "
            f"Scope: {cfg['fine_tuning_scope']})..."
        )

        train_df = df_data.iloc[train_idx]
        val_df = df_data.iloc[val_idx]

        for row_index in train_idx:
            split_assignment_rows.append({
                "filepath": df_data.iloc[row_index]["filepath"],
                "label": int(df_data.iloc[row_index]["label"]),
                "group": df_data.iloc[row_index]["group"],
                "fold": fold,
                "partition": "train",
            })
        for row_index in val_idx:
            split_assignment_rows.append({
                "filepath": df_data.iloc[row_index]["filepath"],
                "label": int(df_data.iloc[row_index]["label"]),
                "group": df_data.iloc[row_index]["group"],
                "fold": fold,
                "partition": "validation",
            })

        train_generator = torch.Generator()
        train_generator.manual_seed(fold_seed)
        val_generator = torch.Generator()
        val_generator.manual_seed(fold_seed + 10000)

        loader_kwargs = {
            "worker_init_fn": seed_worker,
            "num_workers": NUM_WORKERS,
            "pin_memory": torch.cuda.is_available(),
        }
        if NUM_WORKERS > 0:
            loader_kwargs["persistent_workers"] = True
            loader_kwargs["prefetch_factor"] = PREFETCH_FACTOR

        train_loader = DataLoader(
            CustomDataset(train_df, train_transform),
            batch_size=cfg["batch_size"],
            shuffle=True,
            generator=train_generator,
            **loader_kwargs,
        )
        val_loader = DataLoader(
            CustomDataset(val_df, validation_transform),
            batch_size=cfg["batch_size"],
            shuffle=False,
            generator=val_generator,
            **loader_kwargs,
        )

        model = timm.create_model(
            MODEL_TO_RUN,
            pretrained=True,
            num_classes=1,
        ).to(DEVICE)

        fold_parameter_statistics = apply_fine_tuning_scope(
            model,
            BACKBONE_NAME,
            cfg["fine_tuning_scope"],
        )


        optimizer = optim.AdamW(
            filter(lambda parameter: parameter.requires_grad, model.parameters()),
            lr=cfg["lr"],
            weight_decay=cfg["weight_decay"],
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=EPOCHS,
        )
        criterion = nn.BCEWithLogitsLoss()

        best_weights = None
        best_auc = float("-inf")
        best_epoch = 0
        best_checkpoint_val_loss = float("nan")
        early_stop_counter = 0
        scaler = create_grad_scaler()

        # Preserve the four learning-curve fields.
        history = {
            "accuracy": [],
            "val_accuracy": [],
            "loss": [],
            "val_loss": [],
        }
        extended_history = []

        for epoch in range(EPOCHS):
            model.train()
            set_frozen_batchnorm_eval(model)

            train_loss_sum = 0.0
            train_correct = 0
            train_total = 0
            lr_used_this_epoch = optimizer.param_groups[0]["lr"]

            for images, labels in train_loader:
                images = images.to(DEVICE)
                labels = labels.to(DEVICE).unsqueeze(1)

                optimizer.zero_grad(set_to_none=True)
                with autocast_context():
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

                train_loss_sum += loss.item() * images.size(0)
                train_correct += (
                    (torch.sigmoid(outputs) > 0.5) == labels
                ).sum().item()
                train_total += images.size(0)

            model.eval()
            val_loss_sum = 0.0
            val_correct = 0
            val_total = 0
            epoch_val_true = []
            epoch_val_prob = []

            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(DEVICE)
                    labels_device = labels.to(DEVICE).unsqueeze(1)
                    with autocast_context():
                        outputs = model(images)
                        validation_loss = criterion(outputs, labels_device)
                    probabilities = torch.sigmoid(outputs)

                    val_loss_sum += (
                        validation_loss.item() * images.size(0)
                    )
                    val_correct += (
                        (probabilities > 0.5) == labels_device
                    ).sum().item()
                    val_total += images.size(0)

                    epoch_val_prob.extend(
                        probabilities.cpu().numpy().flatten()
                    )
                    epoch_val_true.extend(labels.numpy())

            val_loss_avg = val_loss_sum / val_total
            train_acc = train_correct / train_total
            val_acc = val_correct / val_total
            train_loss_avg = train_loss_sum / train_total

            try:
                val_auc_epoch = roc_auc_score(
                    np.asarray(epoch_val_true),
                    np.asarray(epoch_val_prob),
                )
            except ValueError:
                val_auc_epoch = np.nan

            history["accuracy"].append(train_acc)
            history["val_accuracy"].append(val_acc)
            history["loss"].append(train_loss_avg)
            history["val_loss"].append(val_loss_avg)

            scheduler.step()
            current_lr = optimizer.param_groups[0]["lr"]

            extended_history.append({
                "epoch": epoch + 1,
                "learning_rate_used": lr_used_this_epoch,
                "learning_rate_after_scheduler_step": current_lr,
                "train_accuracy": train_acc,
                "validation_accuracy": val_acc,
                "train_loss": train_loss_avg,
                "validation_loss": val_loss_avg,
                "validation_roc_auc": val_auc_epoch,
                "early_stop_counter_before_update": early_stop_counter,
            })

            print(
                f"Epoch [{epoch + 1:02d}/{EPOCHS}] | "
                f"LR: {current_lr:.2e} | "
                f"Train Acc: {train_acc:.4f} | "
                f"Val Acc: {val_acc:.4f} | "
                f"Val AUC: {val_auc_epoch:.4f}"
            )

            if val_auc_epoch > best_auc + 1e-12:
                best_auc = float(val_auc_epoch)
                best_epoch = epoch + 1
                best_checkpoint_val_loss = float(val_loss_avg)
                best_weights = copy.deepcopy(model.state_dict())
                early_stop_counter = 0
            else:
                early_stop_counter += 1

            if early_stop_counter >= EARLY_STOP_PATIENCE:
                break

        if best_weights is None:
            raise RuntimeError(
                f"Fold {fold} did not produce a valid checkpoint."
            )

        model.load_state_dict(best_weights)
        model.eval()

        y_true = []
        y_prob = []
        with torch.no_grad():
            for images, labels in val_loader:
                with autocast_context():
                    outputs = model(images.to(DEVICE))
                probabilities = torch.sigmoid(outputs)
                y_prob.extend(
                    probabilities.cpu().numpy().flatten()
                )
                y_true.extend(labels.numpy())

        y_true = np.asarray(y_true)
        y_prob = np.asarray(y_prob)

        # Save per-fold outputs.
        prefix = f"{MODEL_TO_RUN}_Fold{fold}"
        roc_auc_value, accuracy_value, f1_value = (
            save_comprehensive_outputs(
                history,
                y_true,
                y_prob,
                prefix,
                CLASS_NAMES,
            )
        )

        # Save additional audit outputs.
        fold_prediction_df = save_fold_additional_outputs(
            fold=fold,
            prefix=prefix,
            train_df=train_df,
            val_df=val_df,
            y_true=y_true,
            y_prob=y_prob,
            extended_history=extended_history,
            parameter_statistics=fold_parameter_statistics,
            best_epoch=best_epoch,
            best_auc=best_auc,
            best_checkpoint_val_loss=best_checkpoint_val_loss,
            fold_seed=fold_seed,
        )

        all_oof_y_true.extend(y_true)
        all_oof_y_prob.extend(y_prob)
        all_oof_records.append(fold_prediction_df)

        # Record fold-level summary fields.
        summary_log.append({
            "Fold": fold,
            "Acc": accuracy_value,
            "Binary_F1_uniform": f1_value,
            "AUC": roc_auc_value,
        })

        fold_detail_rows.append({
            "Fold": fold,
            "Backbone": BACKBONE_NAME,
            "timm_model_name": MODEL_TO_RUN,
            "fine_tuning_scope": cfg["fine_tuning_scope"],
            "trainable_parameter_ratio": fold_parameter_statistics[
                "trainable_parameter_ratio"
            ],
            "trainable_params": fold_parameter_statistics[
                "trainable_params"
            ],
            "total_params": fold_parameter_statistics["total_params"],
            "train_samples": len(train_df),
            "validation_samples": len(val_df),
            "train_groups": train_df["group"].nunique(),
            "validation_groups": val_df["group"].nunique(),
            "validation_phase_samples": int(
                (val_df["label"] == CLASS_TO_IDX["phase"]).sum()
            ),
            "validation_uniform_samples": int(
                (val_df["label"] == CLASS_TO_IDX["uniform"]).sum()
            ),
            "epochs_completed": len(history["loss"]),
            "best_validation_auc_epoch": best_epoch,
            "best_validation_auc": best_auc,
            "validation_loss_at_best_auc": best_checkpoint_val_loss,
            "fold_seed": fold_seed,
            "precision": PRECISION_DESCRIPTION,
            "Accuracy_T0.5": accuracy_value,
            "Binary_F1_uniform_T0.5": f1_value,
            "ROC_AUC": roc_auc_value,
        })

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    # =========================================================================
    # Pool OOF predictions and select the F1-maximizing threshold.
    # =========================================================================
    all_oof_y_true = np.asarray(all_oof_y_true)
    all_oof_y_prob = np.asarray(all_oof_y_prob)

    thresholds = np.arange(0.1, 0.9, 0.01)
    best_threshold = 0.5
    best_f1 = 0.0
    threshold_scan_rows = []

    for threshold in thresholds:
        y_pred = (all_oof_y_prob > threshold).astype(int)
        f1_value = f1_score(all_oof_y_true, y_pred)
        accuracy_value = accuracy_score(all_oof_y_true, y_pred)
        precision_value, recall_value, _, _ = (
            precision_recall_fscore_support(
                all_oof_y_true,
                y_pred,
                average="binary",
            )
        )

        threshold_scan_rows.append({
            "Threshold": threshold,
            "Accuracy": accuracy_value,
            "Binary_Precision_uniform": precision_value,
            "Binary_Recall_uniform": recall_value,
            "Binary_F1_uniform": f1_value,
        })

        if f1_value > best_f1:
            best_f1 = f1_value
            best_threshold = threshold

    final_pred = (all_oof_y_prob > best_threshold).astype(int)
    final_acc = accuracy_score(all_oof_y_true, final_pred)
    final_auc = auc(
        *roc_curve(all_oof_y_true, all_oof_y_prob)[:2]
    )

    df_summary = pd.DataFrame(summary_log)
    df_summary.loc[len(df_summary)] = [
        "Pooled_OOF",
        final_acc,
        best_f1,
        final_auc,
    ]
    stats_std = df_summary.iloc[:-1][
        ["Acc", "Binary_F1_uniform", "AUC"]
    ].std()
    df_summary.loc[len(df_summary)] = [
        "Std",
        stats_std["Acc"],
        stats_std["Binary_F1_uniform"],
        stats_std["AUC"],
    ]
    df_summary["Best_Threshold"] = best_threshold

    # Save the summary table.
    df_summary.to_csv(
        f"CV_SUMMARY_{MODEL_TO_RUN}.csv",
        index=False,
    )

    # =========================================================================
    # Save pooled and audit outputs.
    # =========================================================================
    pd.DataFrame(fold_detail_rows).to_csv(
        f"CV_FOLD_DETAILS_{MODEL_TO_RUN}.csv",
        index=False,
    )
    pd.DataFrame(split_assignment_rows).to_csv(
        f"CV_SPLIT_ASSIGNMENTS_{MODEL_TO_RUN}.csv",
        index=False,
    )

    save_oof_additional_outputs(
        all_oof_y_true=all_oof_y_true,
        all_oof_y_prob=all_oof_y_prob,
        all_oof_records=all_oof_records,
        threshold_scan_df=pd.DataFrame(threshold_scan_rows),
        best_threshold=best_threshold,
    )

    print(
        "\nCross-validation complete. Outputs were saved to the working directory."
    )


if __name__ == "__main__":
    train_and_evaluate_cv()