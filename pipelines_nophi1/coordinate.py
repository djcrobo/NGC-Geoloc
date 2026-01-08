# -*- coding: utf-8 -*-
"""Step① (no phi1 csv):  

This variant removes dependency on CSV phi1 by estimating a similarity
transform parameters (scale, rotation, translation) directly from the
predicted points using non-linear least squares with a good initialization
from an affine estimate.
"""
from __future__ import annotations
import logging, re
from pathlib import Path
import numpy as np, torch, cv2
from scipy.optimize import least_squares


def prepare_points(tensor: torch.Tensor) -> np.ndarray:
    if tensor.dim() == 4 and tensor.size(0) == 1:
        tensor = tensor.squeeze(0)
    return tensor.permute(1, 2, 0).reshape(-1, 2).cpu().numpy().astype(np.float32)


def affine_from_similarity(p):
    s, theta, tx, ty = p
    c, s_ = np.cos(theta), np.sin(theta)
    return np.array([[s * c, -s * s_, tx], [s * s_, s * c, ty]], np.float32)


def transform(A, pts):
    ones = np.ones((pts.shape[0], 1), np.float32)
    return (A @ np.hstack([pts, ones]).T).T


def residuals_sim(p, src, dst):
    return (transform(affine_from_similarity(p), src) - dst).ravel()


def make_src_grid(H: int, W: int, step: int = 8) -> np.ndarray:
    grid = torch.zeros((2, H, W))
    for x in range(W):
        for y in range(H):
            grid[0, y, x] = x * step + step // 2
            grid[1, y, x] = y * step + step // 2
    return prepare_points(grid)


def _init_from_affine(A0: np.ndarray):
    # A0: 2x3, extract similarity params
    M = A0[:, :2]
    tx, ty = A0[0, 2], A0[1, 2]
    # scale approximated by norm of first column (robust for similarity)
    s0 = float(np.linalg.norm(M[:, 0]))
    theta0 = float(np.arctan2(M[1, 0], M[0, 0]))
    return [s0, theta0, tx, ty]


def gen_initial_corners_nophi1(pt_dir, out_dir, *, grid_size=(56, 84), pixel_step=8, logger=None):
    log = logger or logging.getLogger(__name__)
    pt_dir, out_dir = map(Path, (pt_dir, out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    src_pts = make_src_grid(*grid_size, step=pixel_step)
    rect = np.array(
        [
            [src_pts[:, 0].min(), src_pts[:, 1].min()],
            [src_pts[:, 0].max(), src_pts[:, 1].min()],
            [src_pts[:, 0].max(), src_pts[:, 1].max()],
            [src_pts[:, 0].min(), src_pts[:, 1].max()],
        ],
        np.float32,
    )

    result = {}
    for pt_file in pt_dir.glob("*.pt"):
        m = re.search(r"(\d+)(?=\.(?:jpg|png)\.pt$)", pt_file.name, re.IGNORECASE)
        if not m:
            continue
        num = int(m.group(1))

        tgt = prepare_points(torch.load(pt_file))
        A0 = cv2.estimateAffine2D(src_pts, tgt, method=cv2.RANSAC)[0]
        if A0 is None:
            continue

        p0 = _init_from_affine(A0)
        opt = least_squares(residuals_sim, p0, args=(src_pts, tgt))
        A = affine_from_similarity(opt.x)
        corners = transform(A, rect).astype(np.float32)
        np.savetxt(out_dir / f"{pt_file.stem}.txt", corners, fmt="%.3f")
        result[num] = corners

    return result


