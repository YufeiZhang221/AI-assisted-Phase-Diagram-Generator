"""Final training and independent test evaluation.

This script trains the selected architecture-aware configuration on 100% of
the development set, saves the final model weights, and evaluates the model
once on an independent test set using a threshold fixed from pooled grouped-CV
out-of-fold predictions. The training, preprocessing, optimization, metric,
and output logic are intentionally preserved in this public script.

Usage
-----
1. Install PyTorch, torchvision, timm, Albumentations, OpenCV,
   scikit-learn, pandas, NumPy, matplotlib, and seaborn.
2. Replace every placeholder in ``BACKBONE_NAME``, ``MANUAL_CONFIG``,
   ``TRAIN_DIR``, and ``TEST_DIR``.
3. Organize both directories as::

       dataset_root/
       ├── phase/
       └── uniform/

4. Set ``epochs`` before inspecting test results, preferably from the median
   best-validation-AUC epoch across grouped-CV folds. Set ``fixed_threshold``
   from pooled OOF predictions only. Never tune either value on the test set.
5. Run ``python final_evaluation.py``. Outputs are written
   to the current working directory.

The required class mapping is ``phase = 0`` and ``uniform = 1``. The positive
class for probability outputs and binary metrics is ``uniform``.
"""

import os
import gc
import cv2
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
from torchvision import datasets
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
    precision_recall_fscore_support,
)
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


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


set_seed(SEED)


# =============================================================================
# 2. Manual configuration
# =============================================================================
# Copy the selected values from the HPO and grouped-CV outputs.
BACKBONE_NAME = "<BACKBONE_NAME>"

MANUAL_CONFIG = {
    "img_size": "<IMAGE_SIZE>",
    "lr": "<LEARNING_RATE>",
    "fine_tuning_scope": "<FINE_TUNING_SCOPE>",
    "batch_size": "<BATCH_SIZE>",
    "weight_decay": "<WEIGHT_DECAY>",

    # Fix this value before examining the independent test set.
    "epochs": "<FINAL_TRAINING_EPOCHS>",

    # Select this threshold from pooled OOF predictions only.
    "fixed_threshold": "<FIXED_OOF_THRESHOLD>",
}

TRAIN_DIR = "<DEVELOPMENT_SET_DIR>"
TEST_DIR = "<INDEPENDENT_TEST_SET_DIR>"

USE_MIXED_PRECISION = True
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP_ENABLED = bool(USE_MIXED_PRECISION and torch.cuda.is_available())
PRECISION_DESCRIPTION = "16-mixed" if AMP_ENABLED else "32-true"

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
        "epochs": MANUAL_CONFIG["epochs"],
        "fixed_threshold": MANUAL_CONFIG["fixed_threshold"],
        "TRAIN_DIR": TRAIN_DIR,
        "TEST_DIR": TEST_DIR,
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

IMG_SIZE = int(MANUAL_CONFIG["img_size"])
EPOCHS = int(MANUAL_CONFIG["epochs"])
FIXED_THRESHOLD = float(MANUAL_CONFIG["fixed_threshold"])


# =============================================================================
# 3. timm model mapping and valid fine-tuning scopes
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

if BACKBONE_NAME not in TIMM_MODEL_NAMES:
    raise ValueError(
        f"Unsupported BACKBONE_NAME: {BACKBONE_NAME}. "
        f"Available models: {list(TIMM_MODEL_NAMES)}"
    )

if MANUAL_CONFIG["fine_tuning_scope"] not in FINE_TUNING_SCOPES[BACKBONE_NAME]:
    raise ValueError(
        f"Invalid fine_tuning_scope "
        f"'{MANUAL_CONFIG['fine_tuning_scope']}' for {BACKBONE_NAME}. "
        f"Valid scopes: {FINE_TUNING_SCOPES[BACKBONE_NAME]}"
    )

MODEL_TO_RUN = TIMM_MODEL_NAMES[BACKBONE_NAME]


# =============================================================================
# 4. Mixed-precision helpers
# =============================================================================
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
# 5. Dataset and unified augmentation
# =============================================================================
class SimpleDataset(Dataset):
    def __init__(self, image_folder, transform=None):
        self.image_folder = image_folder
        self.transform = transform

    def __len__(self):
        return len(self.image_folder.samples)

    def __getitem__(self, idx):
        path, label = self.image_folder.samples[idx]

        # Identical to the architecture-aware HPO and grouped-CV pipeline:
        # explicitly read a single channel, then replicate it to RGB.
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Failed to read image: {path}")

        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        if self.transform is not None:
            image = self.transform(image=image)["image"]

        return image, torch.tensor(label, dtype=torch.float32)


def get_transforms():
    # Must remain identical to the HPO and grouped-CV augmentation pipeline.
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

    test_transform = A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
        ToTensorV2(),
    ])

    return train_transform, test_transform


# =============================================================================
# 6. Architecture-aware fine-tuning
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
    Freeze the complete model first, then unfreeze complete architecture-aware
    units. This implementation is identical to the HPO and grouped-CV scripts.
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

    if backbone_name.startswith("ResNet"):
        set_module_trainable(model.fc, trainable=True)

        if fine_tuning_scope == "head_only":
            pass
        elif fine_tuning_scope == "layer4_last1":
            set_module_trainable(model.layer4[-1], trainable=True)
        elif fine_tuning_scope == "layer4":
            set_module_trainable(model.layer4, trainable=True)
        elif fine_tuning_scope == "layer3_last1_plus_layer4":
            unfreeze_modules([model.layer3[-1], model.layer4])
        elif fine_tuning_scope == "layer3_last3_plus_layer4":
            unfreeze_modules([
                *list(model.layer3.children())[-3:],
                model.layer4,
            ])
        elif fine_tuning_scope == "layer3_plus_layer4":
            unfreeze_modules([model.layer3, model.layer4])

    elif backbone_name == "DenseNet121":
        denseblock4_layers = list(model.features.denseblock4.children())
        if len(denseblock4_layers) != 16:
            raise RuntimeError(
                "Unexpected timm DenseNet121 denseblock4 structure: "
                f"expected 16 dense layers, got {len(denseblock4_layers)}."
            )

        unfreeze_modules([model.features.norm5, model.classifier])

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

    elif backbone_name == "EfficientNetB0":
        if len(model.blocks) != 7:
            raise RuntimeError(
                "Unexpected timm EfficientNetB0 block structure: "
                f"expected 7 stages, got {len(model.blocks)}."
            )

        unfreeze_modules([model.conv_head, model.bn2, model.classifier])

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

    elif backbone_name == "MobileNetV2":
        if len(model.blocks) != 7:
            raise RuntimeError(
                "Unexpected timm MobileNetV2 block structure: "
                f"expected 7 stages, got {len(model.blocks)}."
            )

        unfreeze_modules([model.conv_head, model.bn2, model.classifier])

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

    elif backbone_name == "VGG16":
        if len(model.features) != 31:
            raise RuntimeError(
                "Unexpected timm VGG16 feature structure: "
                f"expected 31 modules, got {len(model.features)}."
            )

        unfreeze_modules([model.pre_logits, model.head])

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
            unfreeze_modules([model.stages[2], model.stages[3]])

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
    Keep running statistics fixed for BatchNorm layers whose affine parameters
    are frozen. This matches the HPO and grouped-CV behavior.
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
# 7. Output helpers
# =============================================================================
def package_version(distribution_name):
    try:
        return importlib_metadata.version(distribution_name)
    except importlib_metadata.PackageNotFoundError:
        return "not installed"


def save_configuration(parameter_statistics, class_names, class_to_idx):
    configuration = {
        "backbone_name": BACKBONE_NAME,
        "timm_model_name": MODEL_TO_RUN,
        "model_implementation": "timm",
        "manual_config": MANUAL_CONFIG,
        "fine_tuning_scope": MANUAL_CONFIG["fine_tuning_scope"],
        "trainable_parameter_ratio": parameter_statistics[
            "trainable_parameter_ratio"
        ],
        "trainable_parameter_percent": 100.0 * parameter_statistics[
            "trainable_parameter_ratio"
        ],
        "trainable_params": parameter_statistics["trainable_params"],
        "total_params": parameter_statistics["total_params"],
        "first_trainable_parameter": parameter_statistics[
            "first_trainable_parameter"
        ],
        "last_trainable_parameter": parameter_statistics[
            "last_trainable_parameter"
        ],
        "training_data_directory": TRAIN_DIR,
        "test_data_directory": TEST_DIR,
        "class_names": class_names,
        "class_to_idx": class_to_idx,
        "positive_class": "uniform",
        "fixed_threshold": FIXED_THRESHOLD,
        "threshold_source": (
            "pooled grouped five-fold out-of-fold predictions; "
            "not tuned on the independent test set"
        ),
        "seed": SEED,
        "precision": PRECISION_DESCRIPTION,
        "augmentation_config_id": AUGMENTATION_CONFIG_ID,
        "device": str(DEVICE),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "timm_version": package_version("timm"),
        "albumentations_version": package_version("albumentations"),
        "opencv_version": cv2.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "scikit_learn_version": package_version("scikit-learn"),
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cudnn_version": (
            torch.backends.cudnn.version()
            if torch.backends.cudnn.is_available()
            else None
        ),
        "gpu_name": (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else None
        ),
    }

    with open(
        f"FINAL_CONFIGURATION_{MODEL_TO_RUN}.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(configuration, file, indent=2, ensure_ascii=False)


def save_training_outputs(history, prefix):
    history_df = pd.DataFrame(history)
    history_df.to_csv(f"{prefix}_TrainingHistory.csv", index=False)

    plt.figure(figsize=(8, 6))
    plt.plot(
        history_df["epoch"],
        history_df["train_loss"],
        linewidth=2,
        label="Train Loss",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Final Training Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{prefix}_TrainingLoss.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.plot(
        history_df["epoch"],
        history_df["train_accuracy"],
        linewidth=2,
        label="Train Accuracy",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Final Training Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{prefix}_TrainingAccuracy.png", dpi=150)
    plt.close()


def save_final_outputs(
    y_true,
    y_prob,
    filepaths,
    prefix,
    class_names,
    parameter_statistics,
):
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    precision_curve, recall_curve, pr_thresholds = (
        precision_recall_curve(y_true, y_prob)
    )
    pr_auc = auc(recall_curve, precision_curve)

    y_pred = (y_prob > FIXED_THRESHOLD).astype(int)

    accuracy = accuracy_score(y_true, y_pred)
    binary_precision, binary_recall, binary_f1, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            average="binary",
            zero_division=0,
        )
    )
    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    prediction_df = pd.DataFrame({
        "filepath": filepaths,
        "filename": [os.path.basename(path) for path in filepaths],
        "y_true": y_true.astype(int),
        "true_class": [
            class_names[int(label)] for label in y_true
        ],
        "p_uniform": y_prob,
        "fixed_threshold": FIXED_THRESHOLD,
        "y_pred": y_pred,
        "predicted_class": [
            class_names[int(label)] for label in y_pred
        ],
        "correct": y_true.astype(int) == y_pred,
    })
    prediction_df.to_csv(f"{prefix}_predictions.csv", index=False)

    # Save the probability-only output.
    prediction_df[["y_true", "p_uniform"]].rename(
        columns={"p_uniform": "y_prob"}
    ).to_csv(f"{prefix}_probabilities.csv", index=False)

    pd.DataFrame({
        "FPR": fpr,
        "TPR": tpr,
        "Threshold": roc_thresholds,
    }).to_csv(f"{prefix}_roc_curve_data.csv", index=False)

    padded_pr_thresholds = np.append(pr_thresholds, 1.0)
    pd.DataFrame({
        "Precision": precision_curve,
        "Recall": recall_curve,
        "Threshold": padded_pr_thresholds,
    }).to_csv(f"{prefix}_pr_curve_data.csv", index=False)

    report_path = f"{prefix}_REPORT.txt"
    with open(report_path, "w", encoding="utf-8") as file:
        file.write("====================================================\n")
        file.write(f"FINAL EVALUATION REPORT: {prefix}\n")
        file.write("====================================================\n")
        file.write("Evaluation Mode: STRICT BLIND TEST\n")
        file.write(f"Backbone: {BACKBONE_NAME}\n")
        file.write(f"timm model: {MODEL_TO_RUN}\n")
        file.write(
            f"Fine-tuning scope: "
            f"{MANUAL_CONFIG['fine_tuning_scope']}\n"
        )
        file.write(
            f"Trainable parameter ratio: "
            f"{parameter_statistics['trainable_parameter_ratio']:.6f}\n"
        )
        file.write(f"Training epochs: {EPOCHS}\n")
        file.write(f"Precision mode: {PRECISION_DESCRIPTION}\n")
        file.write(f"Fixed Threshold: {FIXED_THRESHOLD}\n")
        file.write(
            "NOTE: Threshold was determined from pooled CV OOF "
            "predictions and was NOT tuned on the test set.\n\n"
        )
        file.write(f"ROC AUC: {roc_auc:.4f}\n")
        file.write(f"PR AUC: {pr_auc:.4f}\n")
        file.write(f"Accuracy: {accuracy:.4f}\n")
        file.write(f"Binary Precision: "
                   f"{binary_precision:.4f}\n")
        file.write(f"Binary Recall: "
                   f"{binary_recall:.4f}\n")
        file.write(f"Binary F1: "
                   f"{binary_f1:.4f}\n")
        file.write(f"Macro-F1: {macro_f1:.4f}\n\n")
        file.write(
            classification_report(
                y_true,
                y_pred,
                target_names=class_names,
                zero_division=0,
            )
        )

    confusion = confusion_matrix(y_true, y_pred)
    pd.DataFrame(
        confusion,
        index=[f"True_{name}" for name in class_names],
        columns=[f"Pred_{name}" for name in class_names],
    ).to_csv(f"{prefix}_ConfusionMatrix_data.csv")

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        confusion,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title(f"Confusion Matrix (T={FIXED_THRESHOLD})")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(f"{prefix}_ConfusionMatrix.png", dpi=150)
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].plot(
        fpr,
        tpr,
        linewidth=2,
        label=f"ROC (AUC = {roc_auc:.4f})",
    )
    axes[0].plot([0, 1], [0, 1], linestyle="--")
    axes[0].set_title("ROC")
    axes[0].set_xlabel("FPR")
    axes[0].set_ylabel("TPR")
    axes[0].legend()

    axes[1].plot(
        recall_curve,
        precision_curve,
        linewidth=2,
        label=f"PR (AUC = {pr_auc:.4f})",
    )
    axes[1].set_title("PR")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(f"{prefix}_ROC_PR.png", dpi=150)
    plt.close()

    misclassified = prediction_df.loc[~prediction_df["correct"]]
    with open(
        f"{prefix}_Misclassified.txt",
        "w",
        encoding="utf-8",
    ) as file:
        file.write(
            f"=== Blind-test misclassified images "
            f"(Threshold={FIXED_THRESHOLD}) ===\n\n"
        )
        for _, row in misclassified.iterrows():
            file.write(
                f"Filename: {row['filename']} | "
                f"True class: {row['true_class']} | "
                f"Predicted class: {row['predicted_class']} | "
                f"P(uniform): {row['p_uniform']:.6f} | "
                f"Full path: {row['filepath']}\n"
            )

    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "accuracy": accuracy,
        "binary_precision_uniform": binary_precision,
        "binary_recall_uniform": binary_recall,
        "binary_f1_uniform": binary_f1,
        "macro_f1": macro_f1,
        "misclassified_count": int((~prediction_df["correct"]).sum()),
    }


# =============================================================================
# 8. Main workflow
# =============================================================================
def train_final_and_test():
    set_seed(SEED)
    train_transform, test_transform = get_transforms()

    full_train_folder = datasets.ImageFolder(TRAIN_DIR)
    test_folder = datasets.ImageFolder(TEST_DIR)

    if full_train_folder.class_to_idx != test_folder.class_to_idx:
        raise ValueError(
            "Train and test class mappings differ. "
            f"Train: {full_train_folder.class_to_idx}; "
            f"Test: {test_folder.class_to_idx}"
        )

    expected_mapping = {"phase": 0, "uniform": 1}
    if full_train_folder.class_to_idx != expected_mapping:
        raise ValueError(
            "Unexpected class mapping. The pipeline requires "
            f"{expected_mapping}, but found "
            f"{full_train_folder.class_to_idx}."
        )

    class_names = full_train_folder.classes

    train_generator = torch.Generator()
    train_generator.manual_seed(SEED)

    test_generator = torch.Generator()
    test_generator.manual_seed(SEED + 10000)

    loader_kwargs = {
        "worker_init_fn": seed_worker,
        "num_workers": NUM_WORKERS,
        "pin_memory": torch.cuda.is_available(),
    }
    if NUM_WORKERS > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = PREFETCH_FACTOR

    train_loader = DataLoader(
        SimpleDataset(full_train_folder, train_transform),
        batch_size=int(MANUAL_CONFIG["batch_size"]),
        shuffle=True,
        generator=train_generator,
        **loader_kwargs,
    )

    test_loader = DataLoader(
        SimpleDataset(test_folder, test_transform),
        batch_size=int(MANUAL_CONFIG["batch_size"]),
        shuffle=False,
        generator=test_generator,
        **loader_kwargs,
    )

    model = timm.create_model(
        MODEL_TO_RUN,
        pretrained=True,
        num_classes=1,
    ).to(DEVICE)

    parameter_statistics = apply_fine_tuning_scope(
        model,
        BACKBONE_NAME,
        MANUAL_CONFIG["fine_tuning_scope"],
    )

    print("\nFinal-training configuration")
    print(f"Backbone: {BACKBONE_NAME}")
    print(f"timm model: {MODEL_TO_RUN}")
    print(
        f"Fine-tuning scope: "
        f"{MANUAL_CONFIG['fine_tuning_scope']}"
    )
    print(
        f"Trainable parameters: "
        f"{parameter_statistics['trainable_params']:,}/"
        f"{parameter_statistics['total_params']:,} "
        f"({parameter_statistics['trainable_parameter_ratio']:.6f})"
    )
    print(f"Image size: {IMG_SIZE}")
    print(f"Batch size: {MANUAL_CONFIG['batch_size']}")
    print(f"Learning rate: {MANUAL_CONFIG['lr']:.15g}")
    print(
        f"Weight decay: "
        f"{MANUAL_CONFIG['weight_decay']:.15g}"
    )
    print(f"Epochs: {EPOCHS}")
    print(f"Fixed threshold: {FIXED_THRESHOLD}")
    print(f"Precision: {PRECISION_DESCRIPTION}")
    print(f"Augmentation: {AUGMENTATION_CONFIG_ID}")

    optimizer = optim.AdamW(
        filter(
            lambda parameter: parameter.requires_grad,
            model.parameters(),
        ),
        lr=float(MANUAL_CONFIG["lr"]),
        weight_decay=float(MANUAL_CONFIG["weight_decay"]),
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS,
    )
    criterion = nn.BCEWithLogitsLoss()
    scaler = create_grad_scaler()

    training_history = []

    print("\nFinal Training on 100% development set...")

    for epoch in range(EPOCHS):
        model.train()
        set_frozen_batchnorm_eval(model)

        train_total = 0
        train_correct = 0
        train_loss_sum = 0.0
        learning_rate_used = optimizer.param_groups[0]["lr"]

        for images, labels in train_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE).unsqueeze(1)

            optimizer.zero_grad(set_to_none=True)

            with autocast_context():
                logits = model(images)
                loss = criterion(logits, labels)

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            probabilities = torch.sigmoid(logits)
            train_loss_sum += loss.item() * images.size(0)
            train_correct += (
                (probabilities > 0.5) == labels
            ).sum().item()
            train_total += images.size(0)

        train_loss = train_loss_sum / train_total
        train_accuracy = train_correct / train_total

        scheduler.step()
        next_learning_rate = optimizer.param_groups[0]["lr"]

        training_history.append({
            "epoch": epoch + 1,
            "learning_rate_used": learning_rate_used,
            "learning_rate_after_scheduler": next_learning_rate,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
        })

        print(
            f"Epoch {epoch + 1:02d}/{EPOCHS} | "
            f"LR: {learning_rate_used:.3e} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_accuracy:.4f}"
        )

    model_path = f"FINAL_MODEL_{MODEL_TO_RUN}.pth"
    torch.save(model.state_dict(), model_path)

    training_prefix = f"FINAL_TRAIN_{MODEL_TO_RUN}"
    save_training_outputs(training_history, training_prefix)

    save_configuration(
        parameter_statistics,
        class_names,
        full_train_folder.class_to_idx,
    )

    print("\nBlind Test Start...")

    model.eval()
    y_true = []
    y_prob = []
    test_filepaths = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)

            with autocast_context():
                logits = model(images)

            probabilities = torch.sigmoid(logits).cpu().numpy().flatten()
            y_prob.extend(probabilities)
            y_true.extend(labels.numpy())

    test_filepaths = [
        path for path, _ in test_folder.samples
    ]

    y_true_np = np.asarray(y_true)
    y_prob_np = np.asarray(y_prob)

    prefix = f"FINAL_TEST_{MODEL_TO_RUN}"
    metrics = save_final_outputs(
        y_true_np,
        y_prob_np,
        test_filepaths,
        prefix,
        class_names,
        parameter_statistics,
    )

    print(
        f"Misclassified image list saved to: "
        f"{prefix}_Misclassified.txt"
    )
    print("======================================")
    print(
        f"AUC: {metrics['roc_auc']:.4f} | "
        f"PR AUC: {metrics['pr_auc']:.4f} | "
        f"Acc: {metrics['accuracy']:.4f} | "
        f"Binary F1 (uniform): "
        f"{metrics['binary_f1_uniform']:.4f} | "
        f"Macro-F1: {metrics['macro_f1']:.4f}"
    )
    print("======================================")

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    train_final_and_test()