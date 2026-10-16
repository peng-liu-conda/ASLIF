"""Command-line interface for precomputed ASLIF branch outputs."""

import argparse

import numpy as np
from PIL import Image

from aslif import compute_alpha, fuse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--maskrcnn-soft", required=True)
    parser.add_argument("--o2-mask", required=True)
    parser.add_argument("--scene", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--m-count", type=int, required=True)
    parser.add_argument("--o-count", type=int, required=True)
    parser.add_argument("--m-confidence", type=float, required=True)
    parser.add_argument("--o-confidence", type=float, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    with np.load(args.maskrcnn_soft, allow_pickle=False) as data:
        m_soft = data["soft_mask"]
    o_binary = np.asarray(Image.open(args.o2_mask).convert("L")) > 0
    alpha = compute_alpha(args.scene, args.m_count, args.o_count,
                          args.m_confidence, args.o_confidence)
    result = fuse(m_soft, o_binary, alpha)
    Image.fromarray(result.astype(np.uint8) * 255).save(args.output)
    print(f"alpha={alpha:.12f}; output={args.output}")


if __name__ == "__main__":
    main()
