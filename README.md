# AI-Assisted High-Throughput Microfluidic Mapping of Phase Diagrams Across Interaction Regimes

This repository contains the image-processing, deep-learning, and phase-boundary reconstruction code associated with the manuscript:

> **AI-assisted high-throughput microfluidic mapping of phase diagrams across interaction regimes**

The computational workflow includes standardized preprocessing of chamber-level confocal images, binary classification of homogeneous and phase-separated states, and support vector machine (SVM)-based reconstruction of continuous two- and three-dimensional phase boundaries.

## Repository contents

```text
.
├── preprocess_images.py              # Confocal-image preprocessing
├── hyperparameter_search.py          # Two-stage Optuna hyperparameter optimization
├── train_groupkfold.py               # Grouped five-fold cross-validation
├── final_evaluation.py               # Final ConvNeXt-Tiny training and test evaluation
├── predict.py                        # Batch inference using a compatible trained model
├── plot_phase_diagram_2D.py          # Two-dimensional SVM phase-boundary fitting
├── plot_phase_diagram_3D.py          # Three-dimensional SVM surface reconstruction
├── requirements.txt                  # Python dependencies
└── README.md
```

The commands below assume the filenames shown above. If the archived scripts use different names, rename the files or update this README consistently before public release.

## Computational environment

The final recorded ConvNeXt-Tiny workflow used:

- Python 3.12.13
- PyTorch 2.10.0+cu128
- timm 1.0.26
- Albumentations 2.0.8
- OpenCV 4.13.0
- NumPy 2.0.2
- pandas 2.3.3
- scikit-learn 1.6.1
- CUDA 12.8
- cuDNN 9.10.2
- NVIDIA Tesla T4 GPU

Hyperparameter optimization additionally used PyTorch Lightning, Optuna, and `optuna-integration`. Grouped cross-validation and final training used native PyTorch training loops. Mixed-precision training was enabled on compatible CUDA hardware; the final ConvNeXt-Tiny run used 16-bit mixed precision.

CPU inference can be supported by `predict.py`, although model training is substantially faster on a CUDA-capable GPU.

## Installation

Create and activate an isolated environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
```

Install the dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For GPU reproduction, install the PyTorch build compatible with the local CUDA driver before installing the remaining packages. Exact numerical identity across machines is not guaranteed because results may depend on GPU architecture, CUDA, cuDNN, and library versions.

## Image preprocessing

Raw confocal TIFF images were standardized using `preprocess_images.py`. The preprocessing workflow performs:

1. percentile clipping using the 2nd and 98th intensity percentiles;
2. linear intensity normalization to the range [0, 1];
3. Gaussian background estimation using a 15 × 15 kernel;
4. weighted background subtraction with a coefficient of 0.3;
5. enhancement of bright structures using a quadratic intensity term;
6. gamma correction with \(\gamma = 0.6\);
7. conversion to 8-bit grayscale PNG;
8. resizing to 256 × 256 pixels using area interpolation.

These standardized PNG files are the expected inputs for model development and inference. In the HPO, grouped cross-validation, final-training, and inference pipelines, each image is:

1. loaded as a single-channel grayscale array;
2. replicated into three identical channels;
3. resized to 224 × 224 pixels;
4. normalized using the ImageNet channel means and standard deviations.

No center crop is used in the final pipeline.

Training augmentation consists of:

- random affine transformation with scaling of 0.95–1.05, translation up to ±5%, and rotation up to ±30° (`p = 0.5`);
- horizontal flipping (`p = 0.5`);
- random brightness and contrast perturbation with limits of 0.2 (`p = 0.5`).

Validation, test, and inference transforms use only resizing, ImageNet normalization, and tensor conversion.

## Dataset organization and class encoding

The implementation uses the following class-folder encoding:

```text
dataset/
├── phase/       # label 0: phase-separated
└── uniform/     # label 1: homogeneous in the manuscript
```

The single output logit is transformed using the sigmoid function and interpreted as:

```text
sigmoid(output) = P(uniform)
```

In the manuscript, the reader-facing term **homogeneous** is used for the class stored in the `uniform` folder.

The fixed operating threshold selected from pooled out-of-fold predictions is 0.21:

```text
P(uniform) > 0.21  -> Homogeneous
P(uniform) <= 0.21 -> Phase-separated
```

This threshold was selected using the development cohort and was not re-optimized on the independent test set. Changing the threshold defines a different operating point and should be reported explicitly.

## Dataset partitions used in the reported workflow

The full image dataset contained 1,287 chamber-level confocal images:

| Partition | Phase-separated | Homogeneous | Total |
|---|---:|---:|---:|
| HPO training subset | 362 | 357 | 719 |
| HPO validation subset | 156 | 153 | 309 |
| Complete development cohort used for grouped CV | 518 | 510 | 1,028 |
| Independent test set | 138 | 121 | 259 |
| Full dataset | 656 | 631 | 1,287 |

The grouped cross-validation cohort contained 21 experimental groups. Experimental-batch identifiers must be preserved so that images from the same batch remain in the same fold.

## Batch inference

`predict.py` accepts either one standardized image or a directory of standardized images. A compatible model checkpoint must be supplied separately because the final checkpoint is not included in the current repository.

The script writes one row per image with the following fields:

- relative image path;
- `probability_uniform`, interpreted as `P(uniform)`;
- `probability_phase`, calculated as `1 - P(uniform)`;
- the fixed decision threshold;
- the predicted class index;
- the predicted class name.

Example:

```bash
python predict.py \
    /path/to/processed_png \
    /path/to/FINAL_MODEL_convnext_tiny.pth \
    prediction_results \
    --threshold 0.21
```

Use `--recursive` when images should also be collected from subdirectories.

Typical outputs are:

```text
prediction_results/
├── predictions.csv
└── prediction_metadata.json
```

The inference workflow must use the same grayscale-to-three-channel conversion, 224 × 224 resizing, and ImageNet normalization used during model development.

## Hyperparameter optimization

`hyperparameter_search.py` performs model-specific, architecture-aware hyperparameter optimization for the following ImageNet-pretrained convolutional backbones implemented using timm:

- VGG16
- ResNet18
- ResNet50
- DenseNet121
- MobileNetV2
- EfficientNetB0
- ConvNeXt-Tiny

Each model uses a single-logit binary classification head. The complete backbone is initially frozen, after which the classification head and one of three prespecified architecture-specific upper-stage or block combinations are made trainable. Fine-tuning scope is therefore searched as a discrete structural configuration, not as a continuous unfreezing ratio.

### Stage 1: configuration screening

For each backbone:

- sampler: Optuna `TPESampler(seed=42)`;
- requested trials: 24;
- learning-rate range: \(1 \times 10^{-5}\) to \(1 \times 10^{-3}\), logarithmic;
- weight-decay range: \(1 \times 10^{-5}\) to \(1 \times 10^{-3}\), logarithmic;
- batch size: 16, 32, or 64;
- fine-tuning scope: one of three architecture-specific options;
- maximum training duration: 12 epochs;
- pruning: `HyperbandPruner(min_resource=2, max_resource=12, reduction_factor=3)`;
- early stopping: validation ROC-AUC, patience 3;
- optimization objective: maximum validation ROC-AUC observed during the trial.

### Stage 2: full-budget confirmation

The three highest-scoring completed Stage-1 configurations are reinitialized from ImageNet-pretrained weights and trained for 25 epochs without pruning or early stopping. The final configuration for each backbone is selected according to Stage-2 validation ROC-AUC.

The independent test set must remain inaccessible throughout HPO.

## Grouped five-fold cross-validation

`train_groupkfold.py` evaluates the selected configuration for each backbone using grouped five-fold cross-validation on the complete 1,028-image development cohort.

Key rules:

- grouping variable: experimental batch;
- number of experimental groups: 21;
- splitter: `GroupKFold(n_splits=5)`;
- images from the same batch must not appear in both training and validation partitions;
- each fold is independently initialized from ImageNet-pretrained weights;
- maximum training duration: 25 epochs;
- early stopping monitor: validation ROC-AUC;
- early stopping patience: 5 epochs;
- best-checkpoint selection: highest validation ROC-AUC;
- fold-specific seeds: 42–46;
- mixed precision: enabled on compatible CUDA hardware.

Fold-specific accuracy, precision, recall, F1, and confusion matrices calculated at `P(uniform) = 0.5` are diagnostic outputs only. They are not used to determine the final operating threshold.

The validation predictions from all five folds are pooled to produce out-of-fold predictions. These pooled predictions are used to:

1. calculate pooled OOF ROC-AUC and PR-AUC;
2. scan thresholds from 0.10 to 0.89 in increments of 0.01;
3. select the threshold maximizing binary F1 for the encoded positive `uniform` class;
4. calculate accuracy, binary F1 for the encoded positive `uniform` class, and macro-F1 at the selected threshold;
5. compare the seven candidate architectures without accessing the independent test set.

For ConvNeXt-Tiny, the selected OOF threshold was 0.21.

## Final ConvNeXt-Tiny training and evaluation

`final_evaluation.py` trains the selected ConvNeXt-Tiny configuration on the complete development cohort and evaluates the resulting model once on the independent held-out test set.

The final configuration was:

| Parameter | Value |
|---|---|
| Backbone | ConvNeXt-Tiny (`convnext_tiny`) |
| Input size | 224 × 224 pixels |
| Fine-tuning scope | `stage3_last8_plus_stage4` |
| Trainable parameters | 25,088,257 / 27,820,897 (90.18%) |
| Batch size | 16 |
| Learning rate | \(2.00 \times 10^{-4}\) |
| Weight decay | \(4.87 \times 10^{-5}\) |
| Final training duration | 3 epochs |
| Fixed threshold | `P(uniform) = 0.21` |
| Seed | 42 |
| Precision | 16-mixed |

The final training duration was prespecified as the median of the fold-specific epochs at which validation ROC-AUC was maximal during grouped cross-validation. Final training used the complete development cohort and did not use a validation subset, early stopping, or validation-based checkpoint selection.

### Independent test performance

The independent test set contained 259 images: 138 phase-separated and 121 homogeneous.

| Metric | Value |
|---|---:|
| ROC-AUC | 0.9956 |
| PR-AUC | 0.9950 |
| Accuracy | 0.9614 |
| Binary F1 (`uniform` as the positive class) | 0.9583 |
| Macro-F1 | 0.9612 |

The confusion matrix contained four phase-separated images classified as homogeneous and six homogeneous images classified as phase-separated.

These values describe performance on the reported held-out test set. They should not be presented as performance on new external datasets.

## Phase-boundary reconstruction

### Two-dimensional phase diagrams

`plot_phase_diagram_2D.py` converts experimentally defined concentration coordinates and binary phase-state labels into continuous two-dimensional phase maps using an RBF-kernel SVM.

The general workflow is:

1. construct a scikit-learn pipeline containing `StandardScaler` followed by an RBF-kernel `SVC`;
2. optimize `C` and `gamma` using three-fold `GridSearchCV`, with feature scaling fitted independently within each cross-validation training fold;
3. enable SVM probability estimation with `random_state = 42`;
4. refit the selected pipeline on the complete concentration-coordinate dataset;
5. evaluate `P(TRUE)` on an 800 × 800 concentration grid;
6. define the phase boundary as the `P(TRUE) = 0.5` contour.

The default grid-search execution uses one CPU job (`--jobs 1`) to avoid uncontrolled process oversubscription. For the phase maps reported in the manuscript, the initial chamber labels were manually assigned. Classifier-derived labels can be passed to the same concentration-coordinate fitting workflow.

### Three-dimensional phase diagrams

`plot_phase_diagram_3D.py` reconstructs the peptide–polyU–PEG8000 phase-boundary surface from three-variable concentration coordinates.

The reported workflow used:

```text
kernel = RBF
C = 5
gamma = 0.3
probability = True
grid size = 80 × 80 × 80
surface definition = decision_function = 0
```

An additional homogeneous constraint point was added at the origin, `[0, 0, 0] -> 0`, to impose physically plausible single-phase behavior at the low-concentration limit. The decision volume was padded at the boundaries, and the continuous surface was extracted using marching cubes. Static plots and interactive HTML visualizations can both be generated.

## Reproducibility notes

- Keep image preprocessing identical across HPO, cross-validation, final training, and inference.
- Preserve the implementation encoding `phase = 0` and `uniform = 1`.
- Interpret the sigmoid output as `P(uniform)`, not `P(phase-separated)`.
- Use the fixed threshold of 0.21 when reproducing the reported final classifier.
- Do not use the independent test set for architecture selection, hyperparameter optimization, threshold tuning, epoch selection, or checkpoint selection.
- Preserve experimental-batch identifiers before grouped cross-validation.
- Control random seeds for Python, NumPy, PyTorch, DataLoader workers, and the Optuna sampler.
- Request deterministic cuDNN behavior and disable cuDNN benchmarking when reproducing the archived workflow.
- Frozen BatchNorm modules should remain in evaluation mode during fine-tuning.
- The classifier provides image-based phase-state assignments. Its predictions are not direct measurements of equilibrium thermodynamic binodals.
- The phase maps reported in the manuscript were constructed from manually annotated images; the trained classifier was evaluated independently as a component for automated future analysis.

## License

A software license will be added before the final public release associated
with the manuscript. Until then, all rights are reserved by the authors.

## Contact

For questions about the code or data, please contact the corresponding authors:

* **Yiwei Li:** [yiweili@hust.edu.cn](mailto:yiweili@hust.edu.cn)
* **Peng Chen:** [gwchenpeng@mail.hust.edu.cn](mailto:gwchenpeng@mail.hust.edu.cn)
* **Bi-Feng Liu:** [bfliu@mail.hust.edu.cn](mailto:bfliu@mail.hust.edu.cn)

**Affiliation:** Key Laboratory for Biomedical Photonics of MOE at Wuhan National Laboratory for Optoelectronics; Hubei Bioinformatics & Molecular Imaging Key Laboratory; Systems Biology Theme; Department of Biomedical Engineering; College of Life Science and Technology; Huazhong University of Science and Technology, Wuhan 430074, China.
