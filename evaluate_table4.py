"""Evaluate Table 4 mask baselines from a CSV manifest.

Required columns: image_id, soft_mask_M_npz_path, binary_mask_O_path,
gt_mask_path. Optional alpha column adds Full ASLIF. Paths may be relative
to the manifest directory. O2SFormer, Union and Intersection are evaluated
without an extra 500-pixel connected-component filter.
"""

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image

from aslif import fuse


def binary(path):
    return np.asarray(Image.open(path).convert("L")) > 0


def metrics(pred, gt):
    tp = int(np.count_nonzero(pred & gt))
    fp = int(np.count_nonzero(pred & ~gt))
    fn = int(np.count_nonzero(~pred & gt))
    return tp, fp, fn


def resolve(base, value):
    path = Path(value)
    return path if path.is_absolute() else base / path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, default=Path("table4_per_image.csv"))
    args = parser.parse_args()
    with args.manifest.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    required = {"image_id", "soft_mask_M_npz_path", "binary_mask_O_path", "gt_mask_path"}
    if not rows or not required.issubset(rows[0]):
        parser.error(f"manifest must contain {sorted(required)}")
    output = []
    for row in rows:
        base = args.manifest.parent
        with np.load(resolve(base, row["soft_mask_M_npz_path"]), allow_pickle=False) as data:
            m_soft = data["soft_mask"].astype(np.float32)
        o = binary(resolve(base, row["binary_mask_O_path"]))
        gt = binary(resolve(base, row["gt_mask_path"]))
        if m_soft.shape != o.shape or o.shape != gt.shape:
            raise ValueError(f"shape mismatch for {row['image_id']}")
        m = m_soft >= 0.31
        predictions = {
            "Mask R-CNN": (m, False),
            "O2SFormer": (o, False),
            "Union": (m | o, False),
            "Intersection": (m & o, False),
        }
        if row.get("alpha", "").strip():
            predictions["Full ASLIF"] = (fuse(m_soft, o, float(row["alpha"])), True)
        for method, (pred, filtered) in predictions.items():
            tp, fp, fn = metrics(pred, gt)
            output.append({"image_id": row["image_id"], "method": method,
                           "cc_500_applied": filtered, "tp": tp, "fp": fp, "fn": fn})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=output[0].keys())
        writer.writeheader()
        writer.writerows(output)
    for method in dict.fromkeys(item["method"] for item in output):
        selected = [item for item in output if item["method"] == method]
        tp, fp, fn = (sum(item[key] for item in selected) for key in ("tp", "fp", "fn"))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
        iou = tp / (tp + fp + fn) if tp + fp + fn else 0.0
        print(f"{method}: n={len(selected)} precision={precision:.6f} recall={recall:.6f} F1={f1:.6f} IoU={iou:.6f}")


if __name__ == "__main__":
    main()
