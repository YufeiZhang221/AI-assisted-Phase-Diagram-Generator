"""Two-stage architecture-aware hyperparameter optimization.

This script performs binary image-classification HPO with a fast Optuna
screening stage followed by full-budget re-evaluation of the top candidates.
The architecture-aware fine-tuning scopes, augmentation pipeline, search
space, optimization settings, reproducibility controls, metrics, and output
files are intentionally preserved in this public script.

Usage
-----
1. Install the required packages: PyTorch, torchvision, timm,
   PyTorch Lightning, Optuna, optuna-integration, Albumentations, OpenCV,
   scikit-learn, NumPy, and pandas.
2. Replace the three placeholders in the configuration sections:
   ``<BACKBONE_NAME>``, ``<HPO_TRAIN_DIR>``, and
   ``<HPO_VALIDATION_DIR>``.
3. Organize both datasets as::

       dataset_root/
       ├── phase/
       └── uniform/

4. Run ``hyperparameter_search.py``. Outputs are written to the
   current working directory.

Supported backbone keys are defined in ``TIMM_MODEL_NAMES``. Images are read
as grayscale, replicated to three channels, and normalized with ImageNet
statistics. The positive class is determined by ImageFolder's alphabetical
mapping and should remain consistent across all pipeline stages.
"""

import gc
import os
import json
import platform
import random
from importlib import metadata as importlib_metadata

import numpy as np
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import optuna
import pandas as pd

# Import PyTorch before Lightning. Some hosted notebook environments may not
# expose the internal torch._utils module as an attribute until it is imported
# explicitly, while Lightning accesses torch._utils during its own import.
import torch
try:
    import torch._utils  # noqa: F401
except Exception as exc:
    raise RuntimeError(
        "PyTorch was imported, but torch._utils could not be loaded. "
        "This usually indicates that the PyTorch installation is corrupted "
        "or shadowed by a local file/folder named 'torch'. "
        f"torch path: {getattr(torch, '__file__', 'unknown')}; "
        f"torch version: {getattr(torch, '__version__', 'unknown')}"
    ) from exc

import torchvision
import timm
import pytorch_lightning as pl
from optuna.integration import PyTorchLightningPruningCallback
from pytorch_lightning.callbacks import EarlyStopping
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets


# ---------------------------------------------------------------------
# 1. Global seed and deterministic settings
# ---------------------------------------------------------------------
SEED = 42
pl.seed_everything(SEED, workers=True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    """Derive NumPy/Python seeds from the DataLoader worker seed."""
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def reset_trial_random_state(trial_seed):
    """
    Reset model-initialization, augmentation, and DataLoader random states at
    the beginning of every trial. The same seed is intentionally used for all
    candidates so that hyperparameter configurations are compared under the
    same controlled stochastic conditions.
    """
    pl.seed_everything(trial_seed, workers=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------
# 2. Global configuration: two-stage Scheme B
# ---------------------------------------------------------------------
BACKBONE_NAME = "<BACKBONE_NAME>"

# Stage 1: fast architecture-aware screening with Optuna + Hyperband.
STAGE1_MAX_EPOCHS = 12
STAGE1_N_TRIALS = 24
STAGE1_EARLY_STOP_PATIENCE = 3
STAGE1_TOP_K = 3

# Stage 2: retrain the top Stage-1 configurations for the full budget.
# No pruning and no early stopping are used in Stage 2.
STAGE2_MAX_EPOCHS = 25

# Limit worker processes to a conservative upper bound for hosted runtimes.
NUM_WORKERS = min(4, os.cpu_count() or 2)
PREFETCH_FACTOR = 2

# Mixed precision substantially reduces GPU time on supported CUDA GPUs.
USE_MIXED_PRECISION = True
TRAINER_PRECISION = (
    "16-mixed"
    if USE_MIXED_PRECISION and torch.cuda.is_available()
    else "32-true"
)

AUGMENTATION_CONFIG_ID = "grayscale_albumentations_v1"
IMG_SIZE = 224


# ---------------------------------------------------------------------
# 3. timm backbone identifiers
# ---------------------------------------------------------------------
# The same timm identifiers should also be used in grouped CV and final
# training so that model definitions, pretrained weights, module names, and
# architecture-aware fine-tuning scopes remain on one implementation chain.
TIMM_MODEL_NAMES = {
    "ResNet18": "resnet18",
    "ResNet50": "resnet50",
    "VGG16": "vgg16",
    "DenseNet121": "densenet121",
    "EfficientNetB0": "efficientnet_b0",
    "MobileNetV2": "mobilenetv2_100",
    "ConvNeXt_Tiny": "convnext_tiny",
}


def build_binary_model(backbone_name, pretrained=True):
    if backbone_name not in TIMM_MODEL_NAMES:
        raise ValueError(
            f"Unsupported backbone: {backbone_name}. "
            f"Available models: {list(TIMM_MODEL_NAMES)}"
        )

    return timm.create_model(
        TIMM_MODEL_NAMES[backbone_name],
        pretrained=pretrained,
        num_classes=1,
    )


# ---------------------------------------------------------------------
# 4. Architecture-aware fine-tuning search spaces
# ---------------------------------------------------------------------
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


# Targeted three-scope search used in Scheme B. Each model is evaluated at one
# shallower scope, the legacy-equivalent scope, and one deeper scope. The full
# FINE_TUNING_SCOPES dictionary remains available for validation and reporting.
TARGETED_FINE_TUNING_SCOPES = {
    "ResNet18": [
        "layer4",
        "layer3_last1_plus_layer4",
        "layer3_plus_layer4",
    ],
    "ResNet50": [
        "layer4",
        "layer3_last3_plus_layer4",
        "layer3_plus_layer4",
    ],
    "DenseNet121": [
        "denseblock4_last8",
        "denseblock4_last13",
        "denseblock4",
    ],
    "EfficientNetB0": [
        "stage5_to_7",
        "stage4_last2_plus_stage5_to_7",
        "stage4_to_7",
    ],
    "MobileNetV2": [
        "stage5_to_7",
        "stage4_to_7",
        "stage3_to_7",
    ],
    "VGG16": [
        "block5",
        "conv4_3_plus_block5",
        "block4_plus_block5",
    ],
    "ConvNeXt_Tiny": [
        "stage3_last6_plus_stage4",
        "stage3_last8_plus_stage4",
        "stage3_plus_stage4",
    ],
}

# ---------------------------------------------------------------------
# 5. Unified Albumentations preprocessing
# ---------------------------------------------------------------------
def get_transforms(img_size=IMG_SIZE):
    train_transform = A.Compose([
        A.Resize(img_size, img_size),
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
        A.Resize(img_size, img_size),
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
        ToTensorV2(),
    ])
    return train_transform, validation_transform


class AlbumentationsImageFolder(Dataset):
    def __init__(self, root, transform=None):
        folder = datasets.ImageFolder(root=root)
        self.samples = folder.samples
        self.targets = folder.targets
        self.classes = folder.classes
        self.class_to_idx = folder.class_to_idx
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Failed to read image: {path}")
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        if self.transform is not None:
            image = self.transform(image=image)["image"]

        return image, torch.tensor(label, dtype=torch.long)


# ---------------------------------------------------------------------
# 6. Manually separated HPO train/validation sets
# ---------------------------------------------------------------------
train_path = "<HPO_TRAIN_DIR>"
val_path = "<HPO_VALIDATION_DIR>"


def validate_public_configuration():
    unresolved = {
        "BACKBONE_NAME": BACKBONE_NAME,
        "train_path": train_path,
        "val_path": val_path,
    }
    unresolved = [
        name
        for name, value in unresolved.items()
        if isinstance(value, str) and value.startswith("<")
    ]
    if unresolved:
        raise ValueError(
            "Replace the public-release placeholders before running: "
            + ", ".join(unresolved)
        )


validate_public_configuration()

train_transform, val_transform = get_transforms(IMG_SIZE)

train_subset = AlbumentationsImageFolder(
    root=train_path,
    transform=train_transform,
)
val_subset = AlbumentationsImageFolder(
    root=val_path,
    transform=val_transform,
)

if train_subset.class_to_idx != val_subset.class_to_idx:
    raise ValueError(
        "Train and validation class mappings are inconsistent: "
        f"train={train_subset.class_to_idx}, "
        f"val={val_subset.class_to_idx}"
    )

print("Class indices:", train_subset.class_to_idx)


# ---------------------------------------------------------------------
# 7. Architecture-aware freezing utilities
# ---------------------------------------------------------------------
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
    units. The task-specific classification head is included in every scope.
    All module paths below correspond to timm implementations.
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

        # timm separates the traditional VGG classifier into pre_logits and
        # head; both are included whenever the classification head is trained.
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


# ---------------------------------------------------------------------
# 8. Export the scope-to-ratio reference table
# ---------------------------------------------------------------------
def export_scope_reference(backbone_name):
    """Export every candidate scope and its exact trainable-parameter ratio."""
    reference_model = build_binary_model(
        backbone_name,
        pretrained=False,
    )

    rows = []
    for scope in FINE_TUNING_SCOPES[backbone_name]:
        statistics = apply_fine_tuning_scope(
            reference_model,
            backbone_name,
            scope,
        )
        rows.append({
            "backbone": backbone_name,
            "timm_model_name": TIMM_MODEL_NAMES[backbone_name],
            "model_implementation": "timm",
            "fine_tuning_scope": scope,
            "total_params": statistics["total_params"],
            "trainable_params": statistics["trainable_params"],
            "trainable_parameter_ratio": statistics[
                "trainable_parameter_ratio"
            ],
            "trainable_parameter_percent": 100.0 * statistics[
                "trainable_parameter_ratio"
            ],
            "first_trainable_parameter": statistics[
                "first_trainable_parameter"
            ],
            "last_trainable_parameter": statistics[
                "last_trainable_parameter"
            ],
        })

    reference_df = pd.DataFrame(rows)
    reference_df.to_csv(
        f"scope_reference_{backbone_name}.csv",
        index=False,
    )

    del reference_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return reference_df


# ---------------------------------------------------------------------
# 9. Lightning model
# ---------------------------------------------------------------------
class HPOModel(pl.LightningModule):
    def __init__(
        self,
        backbone_name,
        lr,
        weight_decay,
        fine_tuning_scope,
        max_epochs_for_scheduler,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = build_binary_model(
            backbone_name,
            pretrained=True,
        )

        self.parameter_statistics = apply_fine_tuning_scope(
            self.model,
            backbone_name,
            fine_tuning_scope,
        )

        self.loss_fn = nn.BCEWithLogitsLoss()
        self.validation_predictions = []
        self.validation_targets = []

        self.best_validation_auc = float("-inf")
        self.best_validation_epoch = -1
        self.final_validation_auc = float("nan")
        self.validation_auc_history = []

    def forward(self, x):
        return self.model(x).squeeze(1)

    def on_train_epoch_start(self):
        """Keep frozen BatchNorm running statistics frozen."""
        for module in self.model.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                affine_parameters = list(module.parameters(recurse=False))
                if affine_parameters and not any(
                    parameter.requires_grad
                    for parameter in affine_parameters
                ):
                    module.eval()

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y.float())
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        probabilities = torch.sigmoid(logits)
        self.validation_predictions.append(probabilities.detach().cpu())
        self.validation_targets.append(y.detach().cpu())

    def on_validation_epoch_end(self):
        predictions = torch.cat(self.validation_predictions).numpy()
        targets = torch.cat(self.validation_targets).numpy()

        try:
            auc_value = float(roc_auc_score(targets, predictions))
        except ValueError:
            auc_value = 0.0

        self.final_validation_auc = auc_value
        self.validation_auc_history.append(auc_value)

        if auc_value > self.best_validation_auc:
            self.best_validation_auc = auc_value
            self.best_validation_epoch = int(self.current_epoch) + 1

        self.log(
            "val_auc",
            auc_value,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )

        self.validation_predictions.clear()
        self.validation_targets.clear()

    def configure_optimizers(self):
        trainable_parameters = [
            parameter
            for parameter in self.parameters()
            if parameter.requires_grad
        ]
        if not trainable_parameters:
            raise RuntimeError("No trainable parameters were passed to AdamW.")

        optimizer = torch.optim.AdamW(
            trainable_parameters,
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(self.hparams.max_epochs_for_scheduler),
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
        }


# ---------------------------------------------------------------------
# 10. Reproducible DataLoader builder
# ---------------------------------------------------------------------
def build_dataloaders(batch_size, seed):
    train_generator = torch.Generator()
    train_generator.manual_seed(seed)

    val_generator = torch.Generator()
    val_generator.manual_seed(seed)

    common_kwargs = {
        "worker_init_fn": seed_worker,
        "num_workers": NUM_WORKERS,
        "pin_memory": torch.cuda.is_available(),
    }
    if NUM_WORKERS > 0:
        common_kwargs["persistent_workers"] = True
        common_kwargs["prefetch_factor"] = PREFETCH_FACTOR

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        generator=train_generator,
        **common_kwargs,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        generator=val_generator,
        **common_kwargs,
    )
    return train_loader, val_loader


# ---------------------------------------------------------------------
# 11. Shared reporting helpers
# ---------------------------------------------------------------------
def model_outcome_dict(model):
    history = [float(value) for value in model.validation_auc_history]
    return {
        "epochs_completed": len(history),
        "best_validation_auc": (
            float(model.best_validation_auc)
            if np.isfinite(model.best_validation_auc)
            else None
        ),
        "best_validation_epoch": int(model.best_validation_epoch),
        "final_validation_auc": (
            float(model.final_validation_auc)
            if np.isfinite(model.final_validation_auc)
            else None
        ),
        "validation_auc_history": history,
    }


def record_stage1_trial_outcomes(trial, model, stopped_early=False):
    outcome = model_outcome_dict(model)
    trial.set_user_attr("search_stage", "stage1_fast_screen")
    trial.set_user_attr("trial_seed", SEED)
    trial.set_user_attr("precision", TRAINER_PRECISION)
    trial.set_user_attr("stage1_max_epochs", STAGE1_MAX_EPOCHS)
    trial.set_user_attr("epochs_completed", outcome["epochs_completed"])
    trial.set_user_attr(
        "best_validation_auc", outcome["best_validation_auc"]
    )
    trial.set_user_attr(
        "best_validation_epoch", outcome["best_validation_epoch"]
    )
    trial.set_user_attr(
        "final_validation_auc", outcome["final_validation_auc"]
    )
    trial.set_user_attr(
        "validation_auc_history", outcome["validation_auc_history"]
    )
    trial.set_user_attr(
        "early_stopping_triggered", bool(stopped_early)
    )
    trial.set_user_attr(
        "objective_definition",
        "maximum validation ROC-AUC across completed Stage-1 epochs",
    )


def attach_parameter_attrs(trial, statistics):
    trial.set_user_attr("model_implementation", "timm")
    trial.set_user_attr(
        "timm_model_name", TIMM_MODEL_NAMES[BACKBONE_NAME]
    )
    trial.set_user_attr("total_params", statistics["total_params"])
    trial.set_user_attr("trainable_params", statistics["trainable_params"])
    trial.set_user_attr(
        "trainable_parameter_ratio",
        statistics["trainable_parameter_ratio"],
    )
    trial.set_user_attr(
        "trainable_parameter_percent",
        100.0 * statistics["trainable_parameter_ratio"],
    )
    trial.set_user_attr(
        "first_trainable_parameter",
        statistics["first_trainable_parameter"],
    )
    trial.set_user_attr(
        "last_trainable_parameter",
        statistics["last_trainable_parameter"],
    )


# ---------------------------------------------------------------------
# 12. Stage 1: fast Optuna screening
# ---------------------------------------------------------------------
def stage1_objective(trial):
    reset_trial_random_state(SEED)

    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    weight_decay = trial.suggest_float(
        "weight_decay", 1e-5, 1e-3, log=True
    )
    fine_tuning_scope = trial.suggest_categorical(
        "fine_tuning_scope",
        TARGETED_FINE_TUNING_SCOPES[BACKBONE_NAME],
    )
    batch_size = trial.suggest_categorical(
        "batch_size", [16, 32, 64]
    )

    train_loader, val_loader = build_dataloaders(batch_size, SEED)
    model = HPOModel(
        BACKBONE_NAME,
        lr,
        weight_decay,
        fine_tuning_scope,
        STAGE1_MAX_EPOCHS,
    )

    statistics = model.parameter_statistics
    attach_parameter_attrs(trial, statistics)

    print(
        f"Stage 1 Trial {trial.number:03d} | "
        f"scope={fine_tuning_scope} | batch={batch_size} | "
        f"lr={lr:.3e} | wd={weight_decay:.3e} | "
        f"trainable={statistics['trainable_parameter_ratio']:.6f}"
    )

    trainer = None
    early_stopping = EarlyStopping(
        monitor="val_auc",
        mode="max",
        patience=STAGE1_EARLY_STOP_PATIENCE,
        min_delta=0.0,
        check_finite=True,
        verbose=False,
    )

    try:
        trainer = pl.Trainer(
            max_epochs=STAGE1_MAX_EPOCHS,
            accelerator="auto",
            devices=1,
            precision=TRAINER_PRECISION,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            deterministic=True,
            num_sanity_val_steps=0,
            callbacks=[
                PyTorchLightningPruningCallback(
                    trial,
                    monitor="val_auc",
                ),
                early_stopping,
            ],
        )
        trainer.fit(model, train_loader, val_loader)

        record_stage1_trial_outcomes(
            trial,
            model,
            stopped_early=bool(early_stopping.stopped_epoch > 0),
        )

        if not np.isfinite(model.best_validation_auc):
            raise RuntimeError(
                "No finite validation ROC-AUC was recorded in Stage 1."
            )
        score = float(model.best_validation_auc)

    except optuna.TrialPruned:
        record_stage1_trial_outcomes(
            trial,
            model,
            stopped_early=False,
        )
        raise

    finally:
        del model, train_loader, val_loader
        if trainer is not None:
            del trainer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return score


# ---------------------------------------------------------------------
# 13. Stage 2: full-budget re-evaluation of top Stage-1 candidates
# ---------------------------------------------------------------------
def evaluate_stage2_candidate(candidate_rank, source_trial):
    reset_trial_random_state(SEED)

    params = source_trial.params
    lr = float(params["lr"])
    weight_decay = float(params["weight_decay"])
    fine_tuning_scope = params["fine_tuning_scope"]
    batch_size = int(params["batch_size"])

    train_loader, val_loader = build_dataloaders(batch_size, SEED)
    model = HPOModel(
        BACKBONE_NAME,
        lr,
        weight_decay,
        fine_tuning_scope,
        STAGE2_MAX_EPOCHS,
    )
    statistics = model.parameter_statistics

    print(
        f"Stage 2 Candidate {candidate_rank}/{STAGE1_TOP_K} | "
        f"source_trial={source_trial.number} | "
        f"scope={fine_tuning_scope} | batch={batch_size} | "
        f"lr={lr:.3e} | wd={weight_decay:.3e}"
    )

    trainer = None
    try:
        trainer = pl.Trainer(
            max_epochs=STAGE2_MAX_EPOCHS,
            accelerator="auto",
            devices=1,
            precision=TRAINER_PRECISION,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            deterministic=True,
            num_sanity_val_steps=0,
            callbacks=[],
        )
        trainer.fit(model, train_loader, val_loader)

        outcome = model_outcome_dict(model)
        if outcome["best_validation_auc"] is None:
            raise RuntimeError(
                "No finite validation ROC-AUC was recorded in Stage 2."
            )

        history_df = pd.DataFrame({
            "epoch": np.arange(1, outcome["epochs_completed"] + 1),
            "validation_auc": outcome["validation_auc_history"],
        })
        history_df.to_csv(
            f"stage2_candidate{candidate_rank}_auc_history_"
            f"{BACKBONE_NAME}.csv",
            index=False,
        )

        row = {
            "candidate_rank_from_stage1": candidate_rank,
            "source_stage1_trial_number": source_trial.number,
            "stage1_best_validation_auc": float(source_trial.value),
            "backbone": BACKBONE_NAME,
            "timm_model_name": TIMM_MODEL_NAMES[BACKBONE_NAME],
            "model_implementation": "timm",
            "fine_tuning_scope": fine_tuning_scope,
            "trainable_parameter_ratio": statistics[
                "trainable_parameter_ratio"
            ],
            "trainable_parameter_percent": 100.0 * statistics[
                "trainable_parameter_ratio"
            ],
            "trainable_params": statistics["trainable_params"],
            "total_params": statistics["total_params"],
            "first_trainable_parameter": statistics[
                "first_trainable_parameter"
            ],
            "last_trainable_parameter": statistics[
                "last_trainable_parameter"
            ],
            "batch_size": batch_size,
            "learning_rate": lr,
            "weight_decay": weight_decay,
            "stage2_max_epochs": STAGE2_MAX_EPOCHS,
            "stage2_epochs_completed": outcome["epochs_completed"],
            "stage2_best_validation_auc": outcome[
                "best_validation_auc"
            ],
            "stage2_best_validation_epoch": outcome[
                "best_validation_epoch"
            ],
            "stage2_final_validation_auc": outcome[
                "final_validation_auc"
            ],
            "stage2_validation_auc_history": json.dumps(
                outcome["validation_auc_history"]
            ),
            "trial_seed": SEED,
            "precision": TRAINER_PRECISION,
            "stage2_pruning": False,
            "stage2_early_stopping": False,
            "selection_basis": (
                "maximum validation ROC-AUC across 25 Stage-2 epochs"
            ),
        }

    finally:
        del model, train_loader, val_loader
        if trainer is not None:
            del trainer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return row


# ---------------------------------------------------------------------
# 14. Reproducibility and dependency reports
# ---------------------------------------------------------------------
def safe_package_version(distribution_name):
    try:
        return importlib_metadata.version(distribution_name)
    except importlib_metadata.PackageNotFoundError:
        return "not installed / unavailable"


def build_environment_report(backbone_name):
    gpu_name = None
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)

    return {
        "pipeline_stage": "two-stage architecture-aware HPO",
        "backbone": backbone_name,
        "model_implementation_library": "timm",
        "timm_model_name": TIMM_MODEL_NAMES[backbone_name],
        "targeted_fine_tuning_scopes": TARGETED_FINE_TUNING_SCOPES[
            backbone_name
        ],
        "dataset_enumeration_library": "torchvision.datasets.ImageFolder",
        "image_loading_library": "OpenCV",
        "augmentation_library": "Albumentations",
        "augmentation_config_id": AUGMENTATION_CONFIG_ID,
        "input_processing": "grayscale read followed by 3-channel replication",
        "training_framework": "PyTorch Lightning",
        "optimization_framework": (
            "Optuna TPE + Hyperband in Stage 1; fixed full-budget "
            "re-evaluation in Stage 2"
        ),
        "validation_metric_library": "scikit-learn",
        "software_versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "timm": getattr(
                timm, "__version__", safe_package_version("timm")
            ),
            "torchvision": torchvision.__version__,
            "albumentations": getattr(
                A, "__version__", safe_package_version("albumentations")
            ),
            "opencv": cv2.__version__,
            "pytorch-lightning": pl.__version__,
            "optuna": optuna.__version__,
            "optuna-integration": safe_package_version(
                "optuna-integration"
            ),
            "scikit-learn": safe_package_version("scikit-learn"),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "hardware": {
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "gpu_name": gpu_name,
        },
        "reproducibility": {
            "base_seed": SEED,
            "sampler": "TPESampler",
            "sampler_seed": SEED,
            "trial_seed_reset": True,
            "new_dataloader_generators_per_run": True,
            "sequential_study_execution": True,
            "deterministic_algorithms_requested": True,
            "precision": TRAINER_PRECISION,
            "note": (
                "Exact bitwise identity can still depend on hardware and "
                "software versions."
            ),
        },
        "stage1": {
            "purpose": "fast screening",
            "n_trials": STAGE1_N_TRIALS,
            "max_epochs": STAGE1_MAX_EPOCHS,
            "early_stopping_patience": STAGE1_EARLY_STOP_PATIENCE,
            "pruner": "HyperbandPruner",
            "pruner_min_resource": 2,
            "pruner_reduction_factor": 3,
            "objective": (
                "maximum validation ROC-AUC across completed epochs"
            ),
        },
        "stage2": {
            "purpose": "full-budget confirmation",
            "top_k_candidates": STAGE1_TOP_K,
            "max_epochs": STAGE2_MAX_EPOCHS,
            "pruning": False,
            "early_stopping": False,
            "objective": (
                "maximum validation ROC-AUC across all 25 epochs"
            ),
        },
        "data_loading": {
            "num_workers": NUM_WORKERS,
            "persistent_workers": NUM_WORKERS > 0,
            "prefetch_factor": (
                PREFETCH_FACTOR if NUM_WORKERS > 0 else None
            ),
            "pin_memory": torch.cuda.is_available(),
        },
        "data": {
            "image_size": IMG_SIZE,
            "augmentation_config_id": AUGMENTATION_CONFIG_ID,
            "train_path": train_path,
            "validation_path": val_path,
            "class_to_idx": train_subset.class_to_idx,
            "train_samples": len(train_subset),
            "validation_samples": len(val_subset),
        },
    }


def export_documentation_files(backbone_name):
    environment_report = build_environment_report(backbone_name)
    with open(
        f"hpo_environment_{backbone_name}.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(environment_report, file, indent=2, ensure_ascii=False)

    readme_text = f"""TWO-STAGE ARCHITECTURE-AWARE HPO NOTES

Backbone: {backbone_name}
Model implementation: timm ({TIMM_MODEL_NAMES[backbone_name]})

Stage 1: fast screening
- Trials: {STAGE1_N_TRIALS}
- Maximum epochs per trial: {STAGE1_MAX_EPOCHS}
- Fine-tuning scopes: {TARGETED_FINE_TUNING_SCOPES[backbone_name]}
- Early-stopping patience: {STAGE1_EARLY_STOP_PATIENCE}
- Hyperband pruning: enabled (min_resource=2, reduction_factor=3)
- Objective: maximum validation ROC-AUC across completed epochs

Stage 2: full-budget confirmation
- Candidates: top {STAGE1_TOP_K} completed Stage-1 trials
- Epochs per candidate: {STAGE2_MAX_EPOCHS}
- Pruning: disabled
- Early stopping: disabled
- Objective: maximum validation ROC-AUC across all Stage-2 epochs

Computation
- Precision: {TRAINER_PRECISION}
- Augmentation: {AUGMENTATION_CONFIG_ID}
- DataLoader workers: {NUM_WORKERS}
- Persistent workers: {NUM_WORKERS > 0}
- TPESampler seed: {SEED}
- The same controlled seed is reset before every Stage-1 trial and Stage-2 candidate.

Interpretation
- The final best_hpo_config file is selected from Stage 2, not directly from Stage 1.
- Report the result as the best-performing configuration among the sampled and re-evaluated candidates, not as a proven global optimum.
- HPO, grouped CV, and final training should use the same Albumentations configuration: {AUGMENTATION_CONFIG_ID}.
- Images are read explicitly as grayscale and replicated to three channels before ImageNet normalization.
"""
    with open(
        f"README_HPO_{backbone_name}.txt",
        "w",
        encoding="utf-8",
    ) as file:
        file.write(readme_text)

    return environment_report


# ---------------------------------------------------------------------
# 15. Run two-stage HPO and export comprehensive reports
# ---------------------------------------------------------------------
if __name__ == "__main__":
    if BACKBONE_NAME not in TIMM_MODEL_NAMES:
        raise ValueError(
            f"Invalid BACKBONE_NAME: {BACKBONE_NAME}. "
            f"Available models: {list(TIMM_MODEL_NAMES)}"
        )

    export_documentation_files(BACKBONE_NAME)

    print("\nAll valid architecture-aware scopes:")
    scope_reference_df = export_scope_reference(BACKBONE_NAME)
    print(
        scope_reference_df[
            [
                "fine_tuning_scope",
                "trainable_parameter_ratio",
                "trainable_parameter_percent",
            ]
        ].to_string(index=False)
    )

    targeted_reference_df = scope_reference_df[
        scope_reference_df["fine_tuning_scope"].isin(
            TARGETED_FINE_TUNING_SCOPES[BACKBONE_NAME]
        )
    ].copy()
    targeted_reference_df.to_csv(
        f"targeted_scope_reference_{BACKBONE_NAME}.csv",
        index=False,
    )

    print("\n" + "=" * 72)
    print(f"STAGE 1: fast screening for {BACKBONE_NAME}")
    print("=" * 72)

    optuna.logging.set_verbosity(optuna.logging.INFO)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
        pruner=optuna.pruners.HyperbandPruner(
            min_resource=2,
            max_resource=STAGE1_MAX_EPOCHS,
            reduction_factor=3,
        ),
    )
    study.optimize(
        stage1_objective,
        n_trials=STAGE1_N_TRIALS,
        n_jobs=1,
        gc_after_trial=True,
    )

    # Export scope coverage so the targeted architecture-aware search remains
    # auditable. TPE sampling is stochastic even with a fixed seed, so the file
    # makes it explicit how many trials each scope received and completed.
    scope_coverage_rows = []
    for scope in TARGETED_FINE_TUNING_SCOPES[BACKBONE_NAME]:
        sampled = [
            trial for trial in study.trials
            if trial.params.get("fine_tuning_scope") == scope
        ]
        completed_for_scope = [
            trial for trial in sampled
            if trial.state == optuna.trial.TrialState.COMPLETE
        ]
        pruned_for_scope = [
            trial for trial in sampled
            if trial.state == optuna.trial.TrialState.PRUNED
        ]
        scope_coverage_rows.append({
            "fine_tuning_scope": scope,
            "sampled_trials": len(sampled),
            "completed_trials": len(completed_for_scope),
            "pruned_trials": len(pruned_for_scope),
            "best_completed_validation_auc": (
                max(trial.value for trial in completed_for_scope)
                if completed_for_scope else np.nan
            ),
        })
    scope_coverage_df = pd.DataFrame(scope_coverage_rows)
    scope_coverage_df.to_csv(
        f"stage1_scope_coverage_{BACKBONE_NAME}.csv",
        index=False,
    )
    print("\nStage-1 scope coverage:")
    print(scope_coverage_df.to_string(index=False))
    uncovered_scopes = scope_coverage_df.loc[
        scope_coverage_df["completed_trials"] == 0,
        "fine_tuning_scope",
    ].tolist()
    if uncovered_scopes:
        print(
            "Warning: no completed Stage-1 trial was available for scopes: "
            + ", ".join(uncovered_scopes)
        )

    completed_trials = [
        trial
        for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE
        and trial.value is not None
        and np.isfinite(trial.value)
    ]
    if not completed_trials:
        raise RuntimeError(
            "No Stage-1 trial completed successfully; Stage 2 cannot run."
        )

    completed_trials.sort(key=lambda trial: trial.value, reverse=True)
    selected_stage1_trials = completed_trials[:STAGE1_TOP_K]

    if len(selected_stage1_trials) < STAGE1_TOP_K:
        print(
            f"Warning: only {len(selected_stage1_trials)} completed Stage-1 "
            f"trials are available; Stage 2 will evaluate all of them."
        )

    stage1_results_df = study.trials_dataframe()
    stage1_results_df.to_csv(
        f"hpo_stage1_{BACKBONE_NAME}.csv",
        index=False,
    )
    # Write a compatibility copy using the legacy output filename.
    stage1_results_df.to_csv(
        f"hpo_{BACKBONE_NAME}.csv",
        index=False,
    )

    stage1_candidates_df = pd.DataFrame([
        {
            "candidate_rank_from_stage1": rank,
            "source_stage1_trial_number": trial.number,
            "stage1_best_validation_auc": trial.value,
            "fine_tuning_scope": trial.params["fine_tuning_scope"],
            "batch_size": trial.params["batch_size"],
            "learning_rate": trial.params["lr"],
            "weight_decay": trial.params["weight_decay"],
            "trainable_parameter_ratio": trial.user_attrs[
                "trainable_parameter_ratio"
            ],
            "stage1_best_validation_epoch": trial.user_attrs[
                "best_validation_epoch"
            ],
            "stage1_final_validation_auc": trial.user_attrs[
                "final_validation_auc"
            ],
            "stage1_epochs_completed": trial.user_attrs[
                "epochs_completed"
            ],
        }
        for rank, trial in enumerate(selected_stage1_trials, start=1)
    ])
    stage1_candidates_df.to_csv(
        f"stage1_top_candidates_{BACKBONE_NAME}.csv",
        index=False,
    )

    print("\nSelected Stage-1 candidates:")
    print(stage1_candidates_df.to_string(index=False))

    print("\n" + "=" * 72)
    print(f"STAGE 2: 25-epoch confirmation for {BACKBONE_NAME}")
    print("=" * 72)

    stage2_rows = []
    for rank, trial in enumerate(selected_stage1_trials, start=1):
        stage2_rows.append(evaluate_stage2_candidate(rank, trial))

    stage2_df = pd.DataFrame(stage2_rows)
    stage2_df = stage2_df.sort_values(
        by=[
            "stage2_best_validation_auc",
            "stage2_final_validation_auc",
        ],
        ascending=[False, False],
    ).reset_index(drop=True)
    stage2_df["final_stage2_rank"] = np.arange(1, len(stage2_df) + 1)
    stage2_df.to_csv(
        f"hpo_stage2_top{len(stage2_df)}_{BACKBONE_NAME}.csv",
        index=False,
    )

    selected = stage2_df.iloc[0]

    print("\n" + "=" * 72)
    print("FINAL SELECTED CONFIGURATION FROM STAGE 2")
    print("=" * 72)
    print("Fine-tuning scope:", selected["fine_tuning_scope"])
    print("Batch size:", int(selected["batch_size"]))
    print("Learning rate:", selected["learning_rate"])
    print("Weight decay:", selected["weight_decay"])
    print(
        "Stage-2 best validation AUC:",
        selected["stage2_best_validation_auc"],
    )
    print(
        "Stage-2 best validation epoch:",
        int(selected["stage2_best_validation_epoch"]),
    )
    print(
        "Trainable parameter ratio:",
        selected["trainable_parameter_ratio"],
    )
    print("=" * 72)

    state_counts = {
        state.name: sum(trial.state == state for trial in study.trials)
        for state in optuna.trial.TrialState
    }

    study_summary_df = pd.DataFrame([{
        "backbone": BACKBONE_NAME,
        "timm_model_name": TIMM_MODEL_NAMES[BACKBONE_NAME],
        "model_implementation": "timm",
        "search_design": "two-stage targeted architecture-aware HPO",
        "targeted_scopes": json.dumps(
            TARGETED_FINE_TUNING_SCOPES[BACKBONE_NAME]
        ),
        "stage1_requested_trials": STAGE1_N_TRIALS,
        "stage1_total_trials_recorded": len(study.trials),
        "stage1_completed_trials": state_counts.get("COMPLETE", 0),
        "stage1_pruned_trials": state_counts.get("PRUNED", 0),
        "stage1_failed_trials": state_counts.get("FAIL", 0),
        "stage1_max_epochs": STAGE1_MAX_EPOCHS,
        "stage1_early_stopping_patience": (
            STAGE1_EARLY_STOP_PATIENCE
        ),
        "stage2_candidates_evaluated": len(stage2_df),
        "stage2_max_epochs": STAGE2_MAX_EPOCHS,
        "stage2_pruning": False,
        "stage2_early_stopping": False,
        "selected_source_stage1_trial_number": int(
            selected["source_stage1_trial_number"]
        ),
        "selected_stage1_candidate_rank": int(
            selected["candidate_rank_from_stage1"]
        ),
        "selected_stage2_rank": int(selected["final_stage2_rank"]),
        "selected_fine_tuning_scope": selected[
            "fine_tuning_scope"
        ],
        "selected_best_validation_auc": selected[
            "stage2_best_validation_auc"
        ],
        "selected_best_validation_epoch": int(
            selected["stage2_best_validation_epoch"]
        ),
        "selected_final_validation_auc": selected[
            "stage2_final_validation_auc"
        ],
        "selection_basis": (
            "best validation ROC-AUC among top Stage-1 candidates "
            "re-evaluated for 25 epochs"
        ),
        "sampler": "TPESampler",
        "sampler_seed": SEED,
        "pruner": "HyperbandPruner (Stage 1 only)",
        "precision": TRAINER_PRECISION,
        "img_size": IMG_SIZE,
        "augmentation_config_id": AUGMENTATION_CONFIG_ID,
    }])
    study_summary_df.to_csv(
        f"hpo_study_summary_{BACKBONE_NAME}.csv",
        index=False,
    )

    best_config_df = pd.DataFrame([{
        "backbone": BACKBONE_NAME,
        "timm_model_name": TIMM_MODEL_NAMES[BACKBONE_NAME],
        "model_implementation": "timm",
        "fine_tuning_scope": selected["fine_tuning_scope"],
        "trainable_parameter_ratio": selected[
            "trainable_parameter_ratio"
        ],
        "trainable_parameter_percent": selected[
            "trainable_parameter_percent"
        ],
        "trainable_params": int(selected["trainable_params"]),
        "total_params": int(selected["total_params"]),
        "batch_size": int(selected["batch_size"]),
        "learning_rate": selected["learning_rate"],
        "weight_decay": selected["weight_decay"],
        "best_validation_auc": selected[
            "stage2_best_validation_auc"
        ],
        "best_validation_epoch": int(
            selected["stage2_best_validation_epoch"]
        ),
        "final_validation_auc": selected[
            "stage2_final_validation_auc"
        ],
        "epochs_completed": int(
            selected["stage2_epochs_completed"]
        ),
        "first_trainable_parameter": selected[
            "first_trainable_parameter"
        ],
        "last_trainable_parameter": selected[
            "last_trainable_parameter"
        ],
        "selection_basis": (
            "best-performing Stage-2 configuration among the top "
            "Stage-1 sampled candidates"
        ),
        "objective_definition": (
            "maximum validation ROC-AUC across 25 Stage-2 epochs"
        ),
        "source_stage1_trial_number": int(
            selected["source_stage1_trial_number"]
        ),
        "stage1_candidate_rank": int(
            selected["candidate_rank_from_stage1"]
        ),
        "trial_seed": SEED,
        "optuna_sampler_seed": SEED,
        "precision": TRAINER_PRECISION,
        "img_size": IMG_SIZE,
        "augmentation_config_id": AUGMENTATION_CONFIG_ID,
    }])
    best_config_df.to_csv(
        f"best_hpo_config_{BACKBONE_NAME}.csv",
        index=False,
    )

    print("\nAll two-stage HPO outputs have been saved.")