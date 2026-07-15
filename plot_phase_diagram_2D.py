"""Construct two-dimensional SVM-fitted phase diagrams from Excel workbooks.

This uploaded version preserves the numerical analysis and plotting settings of
the supplied original script while replacing machine-specific paths with
command-line arguments.

The default workflow:
- identifies Excel files whose names contain "Figure";
- locates the worksheet row containing the "feature" header;
- uses peptide concentration and PolyU concentration as the two coordinates;
- applies z-score standardization inside each cross-validation fold;
- selects an RBF-SVM by three-fold GridSearchCV using a leakage-free pipeline;
- evaluates P(TRUE) on an 800 x 800 grid; and
- defines the displayed boundary at P(TRUE) = 0.5.

Example
-------
python plot_phase_diagram_2D.py 2d_phase_diagram_data/ results/
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC


LOGGER = logging.getLogger(__name__)


# =====================================================
# Arial
# =====================================================
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial"]
plt.rcParams["axes.unicode_minus"] = False


def find_header_row(
    path: Path,
    sheet_name: str,
    max_rows: int = 20,
) -> int:
    """Locate the first row containing an exact ``feature`` entry."""
    preview = pd.read_excel(
        path,
        header=None,
        sheet_name=sheet_name,
    )

    for index in range(min(len(preview), max_rows)):
        row_values = (
            preview.iloc[index]
            .astype(str)
            .str.lower()
            .tolist()
        )
        normalized = [
            str(value).strip()
            for value in row_values
        ]
        if "feature" in normalized:
            return index

    raise ValueError("No row containing the 'feature' header was found.")


def find_x_column(
    columns: Sequence[object],
    prefix: str,
) -> str:
    """Match the x-axis column using the original startswith rule."""
    for column in columns:
        column_name = str(column).strip()
        if column_name.startswith(prefix):
            return column_name

    raise KeyError(
        f"No x-axis column started with {prefix!r}; "
        f"observed columns: {[str(column).strip() for column in columns]}"
    )


def require_exact_column(
    columns: Sequence[object],
    requested: str,
) -> str:
    """Require the exact stripped column name used by the original script."""
    stripped_columns = [
        str(column).strip()
        for column in columns
    ]

    if requested not in stripped_columns:
        raise KeyError(
            f"Required column {requested!r} was not found; "
            f"observed columns: {stripped_columns}"
        )

    return requested


def analyze_workbook(
    path: Path,
    output_dir: Path,
    *,
    sheet_name: str,
    x_column_prefix: str,
    y_column: str,
    phase_column: str,
    grid_size: int,
    cv_folds: int,
    random_seed: int,
    jobs: int,
    transparent: bool,
) -> dict[str, object]:
    """Analyze one workbook without changing the original fitting procedure."""
    LOGGER.info("Analyzing %s", path.name)

    # =====================================================
    # 1. Locate header
    # =====================================================
    header_row = find_header_row(
        path,
        sheet_name,
    )

    # =====================================================
    # 2. Read and clean
    # =====================================================
    frame = pd.read_excel(
        path,
        header=header_row,
        sheet_name=sheet_name,
    )

    frame.columns = [
        str(column).strip()
        for column in frame.columns
    ]

    x_name = find_x_column(
        frame.columns,
        x_column_prefix,
    )
    y_name = require_exact_column(
        frame.columns,
        y_column,
    )
    phase_name = require_exact_column(
        frame.columns,
        phase_column,
    )

    frame = frame.dropna(
        subset=[
            x_name,
            y_name,
            phase_name,
        ]
    )

    features = frame[
        [x_name, y_name]
    ].values

    raw_labels = (
        frame[phase_name]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    encoder = LabelEncoder()
    labels = encoder.fit_transform(
        raw_labels
    )

    class_mapping = dict(
        zip(
            encoder.classes_,
            encoder.transform(
                encoder.classes_
            ),
        )
    )

    # Preserve the original fallback behavior.
    true_index = class_mapping.get(
        "TRUE",
        1,
    )

    if len(np.unique(labels)) < 2:
        raise ValueError(
            "At least two phase classes are required."
        )

    # =====================================================
    # 3. Leakage-free parameter selection
    # =====================================================
    # StandardScaler is placed inside the sklearn Pipeline so that each
    # GridSearchCV training fold learns its own scaling parameters. The
    # validation fold therefore does not influence feature standardization.
    model_pipeline = Pipeline([
        (
            "scaler",
            StandardScaler(),
        ),
        (
            "svc",
            SVC(
                probability=True,
                random_state=random_seed,
            ),
        ),
    ])

    parameter_grid = {
        "C": [1, 10, 50, 100],
        "gamma": [1, 0.1, 0.01],
        "kernel": ["rbf"],
    }

    search_parameter_grid = {
        "svc__C": parameter_grid["C"],
        "svc__gamma": parameter_grid["gamma"],
        "svc__kernel": parameter_grid["kernel"],
    }

    search = GridSearchCV(
        estimator=model_pipeline,
        param_grid=search_parameter_grid,
        cv=cv_folds,
        n_jobs=jobs,
    )

    # Raw concentration features are passed to GridSearchCV. Scaling now occurs
    # independently inside every cross-validation split.
    search.fit(
        features,
        labels,
    )

    classifier = search.best_estimator_

    best_parameters = {
        name.removeprefix("svc__"): value
        for name, value in search.best_params_.items()
    }

    LOGGER.info(
        "Best parameters for %s: %s",
        path.name,
        best_parameters,
    )

    # =====================================================
    # 4. Prediction grid
    # =====================================================
    view_x_min = features[:, 0].min() - 1
    view_x_max = features[:, 0].max() + 2

    view_y_min = features[:, 1].min() - 1
    view_y_max = features[:, 1].max() + 2

    xx, yy = np.meshgrid(
        np.linspace(
            view_x_min - 2,
            view_x_max + 2,
            grid_size,
        ),
        np.linspace(
            view_y_min - 2,
            view_y_max + 2,
            grid_size,
        ),
    )

    grid_points = np.c_[
        xx.ravel(),
        yy.ravel(),
    ]

    # The fitted Pipeline applies its training-data scaler before SVC
    # probability estimation.
    probabilities = (
        classifier.predict_proba(
            grid_points
        )[:, true_index]
        .reshape(xx.shape)
    )

    # =====================================================
    # 5. Plotting
    # =====================================================
    figure = plt.figure(
        figsize=(6.3, 5.4),
        dpi=200,
        facecolor="none" if transparent else "white",
    )

    axis = plt.gca()
    axis.set_facecolor(
        "none" if transparent else "white"
    )

    filled_contour = plt.contourf(
        xx,
        yy,
        probabilities,
        levels=100,
        cmap="RdBu_r",
        alpha=0.6,
        antialiased=True,
    )

    colorbar = plt.colorbar(
        filled_contour
    )

    colorbar.set_label(
        "Probability $P(Phase)$",
        fontsize=16,
        fontweight="bold",
    )

    colorbar.ax.tick_params(
        axis="y",
        labelsize=14,
        width=1.8,
        length=5,
    )

    colorbar.outline.set_linewidth(
        1.8
    )

    plt.contour(
        xx,
        yy,
        probabilities,
        levels=[0.5],
        colors="black",
        linewidths=1.5,
        linestyles="--",
    )

    colors = {
        "TRUE": "#b22222",
        "FALSE": "#104e8b",
    }

    for label in ["TRUE", "FALSE"]:
        mask = raw_labels == label

        if np.any(mask):
            plt.scatter(
                features[mask, 0],
                features[mask, 1],
                c=colors[label],
                edgecolors="black",
                linewidths=0.8,
                s=70,
                label=f" {label}",
                zorder=3,
            )

    plt.xlim(
        view_x_min,
        view_x_max,
    )

    plt.ylim(
        view_y_min,
        view_y_max,
    )

    plt.xlabel(
        x_name,
        fontsize=18,
        fontweight="bold",
    )

    plt.ylabel(
        y_name,
        fontsize=18,
        fontweight="bold",
    )

    plt.title(
        f"Phase Diagram Analysis of {path.name}",
        fontsize=14,
    )

    axis.spines["left"].set_linewidth(2.0)
    axis.spines["bottom"].set_linewidth(2.0)
    axis.spines["top"].set_linewidth(2.0)
    axis.spines["right"].set_linewidth(2.0)

    axis.tick_params(
        axis="both",
        which="major",
        labelsize=15,
        width=2.0,
        length=6,
    )

    plt.grid(
        True,
        linestyle=":",
        alpha=0.4,
    )

    output_path = (
        output_dir
        / path.name.replace(
            ".xlsx",
            "_trans.png",
        )
    )

    # The original figure DPI is inherited from plt.figure(dpi=200).
    # No extra tight_layout call or save-time DPI override is introduced.
    plt.savefig(
        output_path,
        bbox_inches="tight",
        transparent=transparent,
    )

    plt.close(
        figure
    )

    LOGGER.info(
        "Saved %s",
        output_path,
    )

    return {
        "source_file": path.name,
        "sheet_name": sheet_name,
        "header_row_zero_based": int(header_row),
        "x_column": x_name,
        "y_column": y_name,
        "phase_column": phase_name,
        "sample_count": int(
            len(labels)
        ),
        "class_mapping": {
            str(name): int(value)
            for name, value in class_mapping.items()
        },
        "probability_column_index": int(
            true_index
        ),
        "parameter_grid": parameter_grid,
        "cross_validation_folds": int(
            cv_folds
        ),
        "best_parameters": (
            best_parameters
        ),
        "best_cross_validation_score": float(
            search.best_score_
        ),
        "grid_shape": [
            int(grid_size),
            int(grid_size),
        ],
        "boundary_definition": (
            "P(TRUE) = 0.5"
        ),
        "model_pipeline": (
            "StandardScaler -> RBF SVC(probability=True)"
        ),
        "random_seed": int(
            random_seed
        ),
        "grid_search_jobs": int(
            jobs
        ),
        "output_file": output_path.name,
        "reproducibility_note": (
            "StandardScaler is fitted independently within each "
            "GridSearchCV training fold through an sklearn Pipeline. "
            "The SVC probability-calibration seed is fixed."
        ),
    }


def parse_args(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    """Parse paths and settings while preserving the original defaults."""
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "input_dir",
        type=Path,
        help=(
            "Directory containing the source Excel workbooks."
        ),
    )

    parser.add_argument(
        "output_dir",
        type=Path,
        help=(
            "Directory used to save figures and metadata."
        ),
    )

    parser.add_argument(
        "--pattern",
        default="*.xlsx",
        help=(
            "Initial workbook glob pattern. Default: *.xlsx."
        ),
    )

    parser.add_argument(
        "--filename-token",
        default="Figure",
        help=(
            "Only workbooks whose filenames contain this token "
            "are analyzed. Default: Figure."
        ),
    )

    parser.add_argument(
        "--sheet",
        default="Sheet1",
        help="Worksheet name. Default: Sheet1.",
    )

    parser.add_argument(
        "--x-column-prefix",
        default="RRASLRRASLRRASL / μM",
        help=(
            "Required prefix of the peptide-concentration column."
        ),
    )

    parser.add_argument(
        "--y-column",
        default="PolyU (mg/L)",
        help=(
            "Exact PolyU-concentration column name."
        ),
    )

    parser.add_argument(
        "--phase-column",
        default="feature",
        help=(
            "Exact binary phase-label column name."
        ),
    )

    parser.add_argument(
        "--grid-size",
        type=int,
        default=800,
        help=(
            "Points per probability-grid axis. Default: 800."
        ),
    )

    parser.add_argument(
        "--cv-folds",
        type=int,
        default=3,
        help=(
            "GridSearchCV fold count. Default: 3."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help=(
            "Random seed used by SVC probability estimation. "
            "Default: 42."
        ),
    )

    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help=(
            "Parallel GridSearchCV jobs. Default: 1. "
            "Use -1 only when unrestricted CPU parallelism is intended."
        ),
    )

    parser.add_argument(
        "--opaque",
        action="store_true",
        help=(
            "Use a white background instead of the original "
            "transparent background."
        ),
    )

    return parser.parse_args(
        argv
    )


def main(
    argv: Sequence[str] | None = None,
) -> int:
    """Analyze all matching source workbooks."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    args = parse_args(
        argv
    )

    input_dir = (
        args.input_dir
        .expanduser()
        .resolve()
    )

    output_dir = (
        args.output_dir
        .expanduser()
        .resolve()
    )

    if not input_dir.is_dir():
        raise NotADirectoryError(
            f"Input directory does not exist: {input_dir}"
        )

    if args.grid_size < 2:
        raise ValueError(
            "grid_size must be at least 2."
        )

    if args.cv_folds < 2:
        raise ValueError(
            "cv_folds must be at least 2."
        )

    if args.jobs == 0:
        raise ValueError(
            "jobs cannot be zero. Use 1 for sequential execution "
            "or -1 for all available CPU cores."
        )

    all_workbooks = list(
        input_dir.glob(
            args.pattern
        )
    )

    workbooks = sorted(
        path
        for path in all_workbooks
        if args.filename_token in path.name
        and not path.name.startswith("~$")
    )

    if not workbooks:
        raise FileNotFoundError(
            "No workbook satisfied the filename and glob filters."
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    summaries: list[
        dict[str, object]
    ] = []

    for workbook in workbooks:
        try:
            summary = analyze_workbook(
                workbook,
                output_dir,
                sheet_name=args.sheet,
                x_column_prefix=args.x_column_prefix,
                y_column=args.y_column,
                phase_column=args.phase_column,
                grid_size=args.grid_size,
                cv_folds=args.cv_folds,
                random_seed=args.seed,
                jobs=args.jobs,
                transparent=not args.opaque,
            )
            summaries.append(
                summary
            )

        except (
            KeyError,
            ValueError,
            FileNotFoundError,
        ) as error:
            LOGGER.warning(
                "Skipping %s: %s",
                workbook.name,
                error,
            )

    if not summaries:
        raise RuntimeError(
            "No workbook could be analyzed successfully."
        )

    metadata_path = (
        output_dir
        / "phase_diagram_2d_summary.json"
    )

    metadata_path.write_text(
        json.dumps(
            summaries,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    LOGGER.info(
        "Completed %d workbook(s).",
        len(summaries),
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
