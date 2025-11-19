from pathlib import Path
import argparse
from collections import Counter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Count frames and sequences based on filename prefixes.")
    parser.add_argument("--images", type=Path, default=Path("images"), help="Directory containing frame images.")
    parser.add_argument("--suffixes", nargs="*", default=[".png", ".jpg", ".jpeg", ".tif", ".tiff"], help="File suffixes to include.")
    parser.add_argument("--prefix-length", type=int, default=7, help="Number of leading characters used as sequence identifier.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.images.exists():
        raise FileNotFoundError(f"Images directory not found: {args.images}")

    suffixes = tuple(s.lower() for s in args.suffixes)
    counter = Counter()
    frame_count = 0

    for path in args.images.glob("*"):
        if path.is_file() and path.suffix.lower() in suffixes:
            frame_count += 1
            seq_id = path.stem[: args.prefix_length]
            counter[seq_id] += 1

    print(f"Total frames: {frame_count}")
    print(f"Unique sequences (prefix length {args.prefix_length}): {len(counter)}")
    print("Top 10 sequences by frame count:")
    for seq, count in counter.most_common(87):
        print(f"  {seq}: {count}")


if __name__ == "__main__":
    main()
