"""Final ASLIF fusion of precomputed branch outputs."""

from __future__ import annotations

import numpy as np
from scipy import ndimage

SCENE_SCORE = {0: 0.2, 1: 0.5, 2: 0.8}
LAMBDA_S = 1.0 / 3.0
LAMBDA_N = 0.5
LAMBDA_C = 1.0 / 6.0


def compute_alpha(scene: int, m_count: int, o_count: int,
                  m_confidence: float, o_confidence: float) -> float:
    """Return O2SFormer weight using the frozen scene, density and confidence cues."""
    if scene not in SCENE_SCORE:
        raise ValueError("scene must be 0, 1 or 2")
    if min(m_count, o_count) < 0:
        raise ValueError("lane counts must be nonnegative")
    if not np.isfinite([m_confidence, o_confidence]).all():
        raise ValueError("confidence values must be finite")
    density = min(max(m_count, o_count) / 6.0, 1.0)
    confidence = np.clip((o_confidence - m_confidence + 1.0) / 2.0, 0.0, 1.0)
    return float(np.clip(LAMBDA_S * SCENE_SCORE[scene] +
                         LAMBDA_N * density + LAMBDA_C * confidence, 0.0, 1.0))


def fuse(m_soft: np.ndarray, o_binary: np.ndarray, alpha: float,
         threshold: float = 0.31, min_area: int = 500) -> np.ndarray:
    """Fuse maps, threshold, then keep 8-connected components of at least min_area."""
    m = np.asarray(m_soft, dtype=np.float32)
    o = np.asarray(o_binary, dtype=bool)
    if m.ndim != 2 or m.shape != o.shape:
        raise ValueError("inputs must be same-size 2D arrays")
    if not np.isfinite(m).all() or np.any((m < 0) | (m > 1)):
        raise ValueError("Mask R-CNN soft map must have finite values in [0, 1]")
    if not 0 <= alpha <= 1 or not 0 <= threshold <= 1 or min_area < 1:
        raise ValueError("invalid alpha, threshold or minimum area")
    raw = ((1 - alpha) * m + alpha * o.astype(np.float32)) >= threshold
    labels, count = ndimage.label(raw, structure=np.ones((3, 3), dtype=np.uint8))
    if count == 0:
        return np.zeros_like(raw)
    areas = np.bincount(labels.ravel())
    keep = areas >= min_area
    keep[0] = False
    return keep[labels]
