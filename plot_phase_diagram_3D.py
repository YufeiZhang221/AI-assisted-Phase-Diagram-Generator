"""Fit and visualize the peptide-PolyU-PEG three-dimensional phase boundary.

The default analysis follows Supplementary Methods S8: z-score feature
standardization, an RBF-SVM with C=5 and gamma=0.3, a single-phase origin
constraint, an 80 x 80 x 80 decision grid, and a decision_function=0 surface.

Example
-------
python plot_phase_diagram_3D.py 3d_phase_diagram_data.xlsx results/
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
from skimage.measure import marching_cubes
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


LOGGER = logging.getLogger(__name__)

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42


def identify_columns(columns: Sequence[object]) -> dict[str, object]:
    """Identify peptide, PolyU, PEG, and phase columns from header text.

    The peptide column is identified by chemically meaningful strings rather
    than by generic concentration units such as "uM", because several columns
    in a phase-diagram source table may contain concentration units.
    """
    mapping: dict[str, object] = {}
    for column in columns:
        normalized = str(column).strip().lower()
        if any(key in normalized for key in ("rras", "peptide", "protein")):
            mapping.setdefault("peptide", column)
        elif "polyu" in normalized or "poly u" in normalized:
            mapping.setdefault("polyu", column)
        elif "peg" in normalized:
            mapping.setdefault("peg", column)
        elif "phase" in normalized or "feature" in normalized:
            mapping.setdefault("phase", column)

    missing = {"peptide", "polyu", "peg", "phase"} - mapping.keys()
    if missing:
        raise ValueError(
            f"Could not identify columns {sorted(missing)}; mapping={mapping}"
        )
    return mapping


def encode_phase_labels(values: pd.Series) -> np.ndarray:
    """Convert common numeric or textual phase labels to {0, 1}.

    Class definition:
        0 = single phase / homogeneous
        1 = phase separated
    """
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().all() and set(numeric.astype(int).unique()).issubset({0, 1}):
        return numeric.astype(int).to_numpy()

    normalized = values.astype(str).str.strip().str.upper()
    label_map = {
        "0": 0,
        "FALSE": 0,
        "NO": 0,
        "UNIFORM": 0,
        "HOMOGENEOUS": 0,
        "SINGLE PHASE": 0,
        "SINGLE-PHASE": 0,
        "1": 1,
        "TRUE": 1,
        "YES": 1,
        "PHASE": 1,
        "PHASE SEPARATED": 1,
        "PHASE-SEPARATED": 1,
        "PHASE SEPARATION": 1,
        "PHASE-SEPARATION": 1,
    }
    unknown = sorted(set(normalized) - label_map.keys())
    if unknown:
        raise ValueError(f"Unrecognized phase labels: {unknown}")
    return normalized.map(label_map).to_numpy(dtype=int)


def pad_axis(axis: np.ndarray) -> np.ndarray:
    """Extend a regularly sampled axis by one point on each side."""
    if len(axis) < 2:
        raise ValueError("At least two grid points are required to pad an axis.")
    step = axis[1] - axis[0]
    return np.concatenate(([axis[0] - step], axis, [axis[-1] + step]))


def fit_phase_surface(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    grid_size: int,
    c_value: float,
    gamma: float,
    random_seed: int,
) -> tuple[SVC, StandardScaler, tuple[np.ndarray, np.ndarray], tuple[np.ndarray, ...]]:
    """Fit the constrained SVM and return its marching-cubes surface."""
    constrained_features = np.vstack((features, np.array([[0.0, 0.0, 0.0]])))
    constrained_labels = np.append(labels, 0)

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(constrained_features)

    classifier = SVC(
        kernel="rbf",
        C=c_value,
        gamma=gamma,
        probability=True,
        random_state=random_seed,
    )
    classifier.fit(scaled_features, constrained_labels)

    axes = tuple(
        np.linspace(features[:, axis].min(), features[:, axis].max(), grid_size)
        for axis in range(3)
    )
    xx, yy, zz = np.meshgrid(*axes, indexing="ij")
    coordinates = np.column_stack((xx.ravel(), yy.ravel(), zz.ravel()))

    decision = classifier.decision_function(
        scaler.transform(coordinates)
    ).reshape(xx.shape)
    if not (decision.min() <= 0.0 <= decision.max()):
        raise RuntimeError(
            "The fitted decision volume does not cross zero; no phase surface exists."
        )

    padded_decision = np.pad(decision, 1, mode="edge")
    padded_axes = tuple(pad_axis(axis) for axis in axes)
    vertices, faces, _, _ = marching_cubes(padded_decision, level=0.0)

    for dimension, axis in enumerate(padded_axes):
        vertices[:, dimension] = axis[0] + (
            vertices[:, dimension] / (len(axis) - 1)
        ) * (axis[-1] - axis[0])

    return classifier, scaler, (vertices, faces), axes


def save_static_figure(
    features: np.ndarray,
    labels: np.ndarray,
    axes: tuple[np.ndarray, ...],
    vertices: np.ndarray,
    faces: np.ndarray,
    column_names: dict[str, object],
    output_path: Path,
    *,
    dpi: int,
    show: bool,
) -> None:
    """Save the static transparent three-dimensional phase diagram."""
    x_axis, y_axis, z_axis = axes

    fig = plt.figure(figsize=(12, 9))
    plot = fig.add_subplot(111, projection="3d")
    fig.patch.set_facecolor("none")
    plot.set_facecolor("none")
    plot.grid(False)

    for axis in (plot.xaxis, plot.yaxis, plot.zaxis):
        axis.pane.fill = False
        axis.pane.set_edgecolor("none")
        # Matplotlib exposes 3D grid styling through _axinfo. This is used only
        # for presentation and does not affect the fitted SVM surface.
        axis._axinfo["grid"]["linewidth"] = 0.0

    plot.plot_trisurf(
        vertices[:, 0],
        vertices[:, 1],
        faces,
        vertices[:, 2],
        color="#90A4AE",
        alpha=0.5,
        linewidth=0.1,
        edgecolor="#5F7783",
        shade=True,
        antialiased=True,
    )

    for peg_value in np.unique(features[:, 2]):
        plane_x, plane_y = np.meshgrid(
            [x_axis.min(), x_axis.max()],
            [y_axis.min(), y_axis.max()],
        )
        plot.plot_surface(
            plane_x,
            plane_y,
            np.full_like(plane_x, peg_value),
            alpha=0.2,
            color="gray",
            shade=False,
            zorder=0,
        )

    plot.scatter(
        features[labels == 0, 0],
        features[labels == 0, 1],
        features[labels == 0, 2],
        color="#377EB8",
        s=60,
        label="Single phase",
        edgecolors="white",
        linewidths=0.3,
    )
    plot.scatter(
        features[labels == 1, 0],
        features[labels == 1, 1],
        features[labels == 1, 2],
        color="#B23648",
        s=60,
        label="Phase separated",
        edgecolors="black",
        linewidths=0.6,
    )

    plot.set(
        xlim=(x_axis.min(), x_axis.max()),
        ylim=(y_axis.min(), y_axis.max()),
        zlim=(z_axis.min(), z_axis.max()),
    )
    plot.set_xlabel(str(column_names["peptide"]), labelpad=15, fontsize=16)
    plot.set_ylabel(str(column_names["polyu"]), labelpad=15, fontsize=16)
    plot.set_zlabel(str(column_names["peg"]), labelpad=15, fontsize=16)
    plot.tick_params(axis="both", which="major", labelsize=12)
    plot.zaxis.set_tick_params(labelsize=12)
    plot.view_init(elev=25, azim=-45)
    plot.legend(loc="upper left", fontsize=14)

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, transparent=True, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def save_interactive_figure(
    features: np.ndarray,
    labels: np.ndarray,
    axes: tuple[np.ndarray, ...],
    vertices: np.ndarray,
    faces: np.ndarray,
    column_names: dict[str, object],
    output_path: Path,
) -> bool:
    """Save an interactive Plotly figure when Plotly is installed."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        LOGGER.warning("Plotly is not installed; interactive HTML was not created.")
        return False

    x_axis, y_axis, _ = axes
    figure = go.Figure()

    figure.add_trace(
        go.Mesh3d(
            x=vertices[:, 0],
            y=vertices[:, 1],
            z=vertices[:, 2],
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            color="#90A4AE",
            opacity=0.5,
            flatshading=False,
            name="Boundary surface",
        )
    )

    for peg_value in np.unique(features[:, 2]):
        figure.add_trace(
            go.Surface(
                x=[x_axis.min(), x_axis.max()],
                y=[y_axis.min(), y_axis.max()],
                z=[[peg_value, peg_value], [peg_value, peg_value]],
                showscale=False,
                opacity=0.25,
                colorscale=[[0, "gray"], [1, "gray"]],
                hoverinfo="skip",
                name=f"PEG level: {peg_value:g}",
            )
        )

    for phase_value, name, color in (
        (0, "Single phase", "#377EB8"),
        (1, "Phase separated", "#B23648"),
    ):
        mask = labels == phase_value
        figure.add_trace(
            go.Scatter3d(
                x=features[mask, 0],
                y=features[mask, 1],
                z=features[mask, 2],
                mode="markers",
                marker={"color": color, "size": 5},
                name=name,
            )
        )

    figure.update_layout(
        font={"size": 16},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        scene={
            "xaxis_title": str(column_names["peptide"]),
            "yaxis_title": str(column_names["polyu"]),
            "zaxis_title": str(column_names["peg"]),
            "xaxis": {"title_font": {"size": 20}, "tickfont": {"size": 14}},
            "yaxis": {"title_font": {"size": 20}, "tickfont": {"size": 14}},
            "zaxis": {"title_font": {"size": 20}, "tickfont": {"size": 14}},
        },
    )
    # Embed Plotly so the deposited HTML remains viewable without network access.
    figure.write_html(output_path, include_plotlyjs=True)
    return True


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse explicit input/output paths and published model parameters."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_file", type=Path, help="Excel workbook containing 3D phase data.")
    parser.add_argument("output_dir", type=Path, help="Directory for static and interactive figures.")
    parser.add_argument("--sheet", default=0, help="Worksheet name or zero-based index.")
    parser.add_argument("--grid-size", type=int, default=80, help="Points per decision-grid axis.")
    parser.add_argument("--c-value", type=float, default=5.0, help="RBF-SVM penalty parameter C.")
    parser.add_argument("--gamma", type=float, default=0.3, help="RBF-SVM gamma value.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for SVC probability estimates.")
    parser.add_argument("--dpi", type=int, default=600, help="Static PNG resolution.")
    parser.add_argument("--skip-html", action="store_true", help="Do not create the interactive Plotly file.")
    parser.add_argument("--show", action="store_true", help="Display the static figure after saving it.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the three-dimensional SVM phase-surface workflow."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args(argv)

    input_path = args.input_file.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if not input_path.is_file():
        raise FileNotFoundError(f"Input workbook does not exist: {input_path}")
    if args.grid_size < 3:
        raise ValueError("Grid size must be at least 3.")

    sheet: str | int = int(args.sheet) if str(args.sheet).isdigit() else args.sheet
    frame = pd.read_excel(input_path, sheet_name=sheet).dropna(axis=1, how="all")

    column_names = identify_columns(frame.columns)
    subset = frame[
        [
            column_names["peptide"],
            column_names["polyu"],
            column_names["peg"],
            column_names["phase"],
        ]
    ].dropna()

    features = subset[
        [column_names["peptide"], column_names["polyu"], column_names["peg"]]
    ].to_numpy(dtype=float)
    labels = encode_phase_labels(subset[column_names["phase"]])

    if len(np.unique(labels)) != 2:
        raise ValueError("Both single-phase and phase-separated observations are required.")

    classifier, _, surface, axes = fit_phase_surface(
        features,
        labels,
        grid_size=args.grid_size,
        c_value=args.c_value,
        gamma=args.gamma,
        random_seed=args.seed,
    )
    vertices, faces = surface

    output_dir.mkdir(parents=True, exist_ok=True)
    static_path = output_dir / "static_3d_phase_right.png"
    save_static_figure(
        features,
        labels,
        axes,
        vertices,
        faces,
        column_names,
        static_path,
        dpi=args.dpi,
        show=args.show,
    )

    html_path = output_dir / "interactive_3d_phase.html"
    html_created = False
    if not args.skip_html:
        html_created = save_interactive_figure(
            features,
            labels,
            axes,
            vertices,
            faces,
            column_names,
            html_path,
        )

    metadata = {
        "source_file": input_path.name,
        "sheet": args.sheet,
        "column_names": {key: str(value) for key, value in column_names.items()},
        "sample_count": int(len(labels)),
        "class_definition": {"0": "single phase", "1": "phase separated"},
        "class_counts": {str(value): int((labels == value).sum()) for value in (0, 1)},
        "feature_ranges": {
            "peptide": [float(features[:, 0].min()), float(features[:, 0].max())],
            "polyu": [float(features[:, 1].min()), float(features[:, 1].max())],
            "peg": [float(features[:, 2].min()), float(features[:, 2].max())],
        },
        "model": {
            "kernel": "rbf",
            "C": args.c_value,
            "gamma": args.gamma,
            "probability": True,
            "random_state": args.seed,
        },
        "preprocessing": "StandardScaler fitted on experimental features plus the origin constraint point",
        "origin_constraint": {"coordinates": [0, 0, 0], "label": 0},
        "grid_shape": [args.grid_size] * 3,
        "boundary": "decision_function = 0",
        "surface": {
            "vertex_count": int(len(vertices)),
            "face_count": int(len(faces)),
            "extraction_method": "skimage.measure.marching_cubes",
            "decision_padding": "np.pad(decision, 1, mode='edge')",
        },
        "support_vector_count": classifier.n_support_.tolist(),
        "static_figure": static_path.name,
        "interactive_figure": html_path.name if html_created else None,
    }
    (output_dir / "phase_diagram_3d_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    LOGGER.info("Saved 3D phase analysis to %s", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
