"""Standardize single-channel confocal TIFF images for classification.

The preprocessing procedure preserves the supplied original implementation:

1. percentile normalization using the 2nd and 98th percentiles;
2. subtraction of 0.3 times a 15 x 15 Gaussian-blurred background;
3. enhancement by adding 0.25 times the squared intensity;
4. gamma correction;
5. direct conversion to 8-bit grayscale;
6. resizing to 256 x 256 pixels using cv2.INTER_AREA;
7. PNG export.

The script assumes that each input TIFF is a two-dimensional, single-channel
confocal image. Existing PNG files are overwritten, matching the behavior of
the original batch-processing script.

Example
-------
python preprocess_images.py raw_tif/ processed_png/ --gamma 0.6
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Sequence

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tifffile


LOGGER = logging.getLogger(__name__)

# Preserve the original file-extension filter.
TIFF_SUFFIX = ".tif"


# ============================================================
# 1. Percentile normalization
# ============================================================

def percentile_normalize(
    image: np.ndarray,
    low_percentile: float = 2,
    high_percentile: float = 98,
) -> np.ndarray:
    """Normalize the selected percentile range to [0, 1]."""

    image = image.astype(
        np.float32
    )

    low, high = np.percentile(
        image,
        (
            low_percentile,
            high_percentile,
        ),
    )

    # Preserve the exact denominator used in the original script.
    image = np.clip(
        (image - low)
        / (
            high
            - low
            + 1e-8
        ),
        0,
        1,
    )

    return image


# ============================================================
# 2. Image enhancement
# ============================================================

def enhance_image(
    image: np.ndarray,
    gamma: float = 0.6,
) -> np.ndarray:
    """Apply the original image-enhancement procedure."""

    # Percentile normalization.
    image = percentile_normalize(
        image
    )

    # Estimate the low-frequency background using the original
    # fixed 15 x 15 Gaussian kernel.
    background = cv2.GaussianBlur(
        image,
        (15, 15),
        0,
    )

    # Preserve the original background-suppression coefficient.
    image = (
        image
        - 0.3
        * background
    )

    image = np.clip(
        image,
        0,
        1,
    )

    # Preserve the original quadratic bright-structure enhancement.
    image = np.clip(
        image
        + 0.25
        * (
            image ** 2
        ),
        0,
        1,
    )

    # Gamma correction.
    image = np.power(
        image,
        gamma,
    )

    # Preserve the original direct float-to-uint8 conversion.
    # Do not use np.rint(), because it changes pixel values.
    image_uint8 = (
        image
        * 255
    ).astype(
        np.uint8
    )

    return image_uint8


# ============================================================
# 3. Optional enhancement preview
# ============================================================

def show_enhancement_compare(
    tif_path: str | Path,
    gamma: float = 0.6,
) -> np.ndarray:
    """Display the original and enhanced images side by side."""

    image = tifffile.imread(
        tif_path
    )

    enhanced = enhance_image(
        image,
        gamma=gamma,
    )

    plt.figure(
        figsize=(10, 5)
    )

    plt.subplot(
        1,
        2,
        1,
    )

    plt.imshow(
        image,
        cmap="gray",
    )

    plt.title(
        "Original image"
    )

    plt.axis(
        "off"
    )

    plt.subplot(
        1,
        2,
        2,
    )

    plt.imshow(
        enhanced,
        cmap="gray",
    )

    plt.title(
        "Enhanced image"
    )

    plt.axis(
        "off"
    )

    plt.tight_layout()
    plt.show()

    return enhanced


# ============================================================
# 4. Process one TIFF
# ============================================================

def process_file(
    input_path: Path,
    output_path: Path,
    *,
    gamma: float,
    size: tuple[int, int],
) -> None:
    """Process one TIFF and save one PNG."""

    image = tifffile.imread(
        input_path
    )

    enhanced = enhance_image(
        image,
        gamma=gamma,
    )

    resized = cv2.resize(
        enhanced,
        size,
        interpolation=cv2.INTER_AREA,
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Existing files are intentionally overwritten, matching the original.
    write_success = cv2.imwrite(
        str(output_path),
        resized,
    )

    if not write_success:
        raise OSError(
            f"OpenCV could not write output image: {output_path}"
        )

    LOGGER.info(
        "Saved: %s",
        output_path,
    )


# ============================================================
# 5. Batch processing
# ============================================================

def batch_process(
    input_dir: Path,
    output_dir: Path,
    *,
    gamma: float = 0.6,
    size: tuple[int, int] = (256, 256),
    recursive: bool = False,
) -> int:
    """Batch-process TIFF images and return the processed file count."""

    input_dir = (
        input_dir
        .expanduser()
        .resolve()
    )

    output_dir = (
        output_dir
        .expanduser()
        .resolve()
    )

    if not input_dir.is_dir():
        raise NotADirectoryError(
            f"Input directory does not exist: {input_dir}"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if recursive:
        iterator = input_dir.rglob(
            "*"
        )
    else:
        iterator = input_dir.iterdir()

    # Do not sort the file list. The original implementation used
    # os.listdir() without explicit sorting.
    input_paths = [
        path
        for path in iterator
        if (
            path.is_file()
            and path.suffix.lower()
            == TIFF_SUFFIX
        )
    ]

    if not input_paths:
        raise FileNotFoundError(
            f"No .tif images were found in {input_dir}"
        )

    for input_path in input_paths:
        if recursive:
            relative_path = input_path.relative_to(
                input_dir
            )

            output_path = (
                output_dir
                / relative_path
            ).with_suffix(
                ".png"
            )

        else:
            output_path = (
                output_dir
                / input_path.name
            ).with_suffix(
                ".png"
            )

        process_file(
            input_path,
            output_path,
            gamma=gamma,
            size=size,
        )

    LOGGER.info(
        "Batch processing completed: %d image(s).",
        len(input_paths),
    )

    return len(
        input_paths
    )


# ============================================================
# 6. Command-line arguments
# ============================================================

def parse_args(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    """Parse paths and preprocessing parameters."""

    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "input_dir",
        type=Path,
        help=(
            "Directory containing the raw single-channel .tif images."
        ),
    )

    parser.add_argument(
        "output_dir",
        type=Path,
        help=(
            "Directory used to save processed PNG images."
        ),
    )

    parser.add_argument(
        "--gamma",
        type=float,
        default=0.6,
        help=(
            "Gamma-correction exponent. "
            "Default: 0.6."
        ),
    )

    parser.add_argument(
        "--width",
        type=int,
        default=256,
        help=(
            "Output width in pixels. "
            "Default: 256."
        ),
    )

    parser.add_argument(
        "--height",
        type=int,
        default=256,
        help=(
            "Output height in pixels. "
            "Default: 256."
        ),
    )

    parser.add_argument(
        "--recursive",
        action="store_true",
        help=(
            "Search input subdirectories recursively. "
            "Disabled by default to preserve the original behavior."
        ),
    )

    return parser.parse_args(
        argv
    )


# ============================================================
# 7. Main workflow
# ============================================================

def main(
    argv: Sequence[str] | None = None,
) -> int:
    """Run TIFF-to-PNG preprocessing."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    args = parse_args(
        argv
    )

    if args.gamma <= 0:
        raise ValueError(
            "gamma must be greater than zero."
        )

    if (
        args.width <= 0
        or args.height <= 0
    ):
        raise ValueError(
            "Output width and height must be positive integers."
        )

    batch_process(
        args.input_dir,
        args.output_dir,
        gamma=args.gamma,
        size=(
            args.width,
            args.height,
        ),
        recursive=args.recursive,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )