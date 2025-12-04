from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np


FILENAME_PATTERN = re.compile(r"^(?P<prefix>[A-Za-z0-9]{6})(?P<frame>\d+)_")


def parse_sequence_key(path: Path) -> Tuple[str, int]:
    """
    Extract the 6-character sequence prefix and numeric frame index from filenames like
    'ns050a1424_resized_512.png'.
    """
    match = FILENAME_PATTERN.match(path.stem)
    if not match:
        raise ValueError(f"Filename does not match expected pattern: {path.name}")
    prefix = match.group("prefix")
    frame_idx = int(match.group("frame"))
    return prefix, frame_idx


def load_grayscale(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return image


def build_diff_dataset(
    images_dir: Path,
    masks_dir: Path,
    out_images_dir: Path,
    out_masks_dir: Path,
    overwrite: bool = False,
) -> Dict[str, int]:
    images_dir = images_dir.resolve()
    masks_dir = masks_dir.resolve()
    out_images_dir.mkdir(parents=True, exist_ok=True)
    out_masks_dir.mkdir(parents=True, exist_ok=True)

    grouped: Dict[str, List[Tuple[int, Path]]] = defaultdict(list)
    for img_path in images_dir.glob("*.png"):
        try:
            prefix, frame_idx = parse_sequence_key(img_path)
        except ValueError:
            continue
        grouped[prefix].append((frame_idx, img_path))

    generated_pairs = 0
    missing_masks = 0
    for prefix, frames in grouped.items():
        frames.sort(key=lambda x: x[0])
        for (idx_a, path_a), (idx_b, path_b) in zip(frames, frames[1:]):
            mask_a_path = masks_dir / path_a.name
            mask_b_path = masks_dir / path_b.name
            if not mask_a_path.exists() or not mask_b_path.exists():
                missing_masks += 1
                continue

            diff_name = f"{prefix}{idx_a:04d}_{prefix}{idx_b:04d}_diff.npy"
            mask_name = f"{prefix}{idx_a:04d}_{prefix}{idx_b:04d}_diff.png"

            diff_path = out_images_dir / diff_name
            mask_out_path = out_masks_dir / mask_name

            if not overwrite and diff_path.exists() and mask_out_path.exists():
                continue

            img_a = load_grayscale(path_a).astype(np.int16)
            img_b = load_grayscale(path_b).astype(np.int16)
            diff = img_a - img_b

            mask_a = load_grayscale(mask_a_path) > 0
            mask_b = load_grayscale(mask_b_path) > 0
            xor_mask = np.logical_xor(mask_a, mask_b).astype(np.uint8) * 255

            np.save(diff_path, diff, allow_pickle=False)
            cv2.imwrite(str(mask_out_path), xor_mask)
            generated_pairs += 1

    return {
        "sequences": len(grouped),
        "pairs": generated_pairs,
        "missing_mask_pairs": missing_masks,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build frame-difference dataset.")
    parser.add_argument("--images", type=Path, default=Path("images"), help="Input frames directory.")
    parser.add_argument("--masks", type=Path, default=Path("masks"), help="Input masks directory.")
    parser.add_argument("--out-images", type=Path, default=Path("d_images"), help="Output directory for diff npy files.")
    parser.add_argument("--out-masks", type=Path, default=Path("d_masks"), help="Output directory for XOR masks.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stats = build_diff_dataset(args.images, args.masks, args.out_images, args.out_masks, args.overwrite)
    print("Dataset generation summary:")
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
