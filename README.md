# ASLIF fusion implementation (revised frozen configuration)

This repository packages the ASLIF fusion and postprocessing used in the final frozen run. It accepts an existing Mask R-CNN soft probability map, an O2SFormer binary mask, scene class, lane counts, and branch confidence scores. It does **not** include model weights, datasets, or upstream model inference.

## Install

```bash
python -m pip install -r requirements.txt
```

## Run

```bash
python run_aslif.py --maskrcnn-soft m.npz --o2-mask o.png --scene 0 --m-count 3 --o-count 4 --m-confidence 0.8 --o-confidence 0.7 --output fused.png
```

`m.npz` must contain a 2D array named `soft_mask` with values in `[0, 1]`. `o.png` must be a same-size binary mask (nonzero means foreground). Scene indices are `0=normal`, `1=multi-lane`, `2=complex`. The output is an 8-bit binary PNG; the command also prints the computed alpha.

The revised frozen parameters are `lambda_s=1/3`, `lambda_n=1/2`, `lambda_c=1/6`, threshold `0.31`, and 8-connected component minimum area `500` pixels, as recorded in `ASLIF_REPAIRED_FINAL_PARAMETERS_BEFORE_TEST300.json`. Alpha weights O2SFormer: `(1-alpha)*M + alpha*O`. The earlier frozen configuration used `(0.272727, 0.409091, 0.318182)` and threshold `0.50`; it is a separate historical run.

## Scope and provenance

The implementation follows the revised run's raw branch confidence convention: mean valid Mask R-CNN instance score and mean finite O2SFormer lane score, or zero when no valid score exists. The manuscript's temperature-scaled confidence description differs from this executable convention; do not claim that this package implements calibrated confidence.

The original experiment also depended on a frozen VAL100 selection set and TEST300 evaluation set. Neither data nor weights are redistributed here. The code in this package is a standalone extraction of the fusion step; published metrics cannot be reproduced from this repository alone. It is **not** the full Mask R-CNN, O2SFormer or scene-classifier training/inference pipeline.

## Table 4 mask evaluation

Run `python evaluate_table4.py manifest.csv --output table4_per_image.csv`. The manifest needs `image_id`, `soft_mask_M_npz_path`, `binary_mask_O_path`, and `gt_mask_path`; optional `alpha` adds Full ASLIF. The script prints micro Precision, Recall, F1 and IoU and saves per-image confusion counts.

For the repaired Table 4 variant in `TABLE4_TEST300_REVISED_REPAIRED_20261008/recompute_table4.py`, Mask R-CNN uses `soft_M >= 0.31`; O2SFormer, Union, and Intersection receive **no additional** 500-pixel connected-component filtering. Full ASLIF does receive it. Another historical Table 4 rebuild used a different common-filter protocol for O-only and Union; results from these protocols must not be mixed.
