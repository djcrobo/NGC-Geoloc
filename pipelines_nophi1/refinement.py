# -*- coding: utf-8 -*-
"""
pipelines_nophi1/refinement.py
-------------------------------------------------
Step③ : 
"""

from __future__ import annotations
import logging, re, tempfile
from pathlib import Path
from typing import Mapping

import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
import torch
from romatch import roma_outdoor


# --------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------- #
def _reproj_err(src: np.ndarray, dst: np.ndarray, H: np.ndarray | None) -> float:
    if H is None:
        return np.inf
    proj = cv2.perspectiveTransform(src.reshape(-1, 1, 2), H).reshape(-1, 2)
    return np.linalg.norm(dst - proj, axis=1).mean()


def _roma_H(query_path: str, ref_path: str, model, device):
    warp, cert = model.match(query_path, ref_path, device=device)
    matches, _ = model.sample(warp, cert)
    Wq, Hq = Image.open(query_path).size
    Wr, Hr = Image.open(ref_path).size
    k1, k2 = model.to_pixel_coordinates(matches, Hq, Wq, Hr, Wr)  # torch.Tensor
    k1_np = k1.cpu().numpy()
    k2_np = k2.cpu().numpy()
    H, _ = cv2.findHomography(k1_np, k2_np, cv2.USAC_MAGSAC, 5.0)
    return H, k1_np, k2_np


def _best_homography(q_path, r_path, model, thresh, device):
    H, k1, k2 = _roma_H(q_path, r_path, model, device)
    err = _reproj_err(k1, k2, H)

    if err <= thresh:
        return H

    # 180° retry
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=True) as tf:
        img = cv2.imread(q_path)
        cv2.imwrite(tf.name, cv2.rotate(img, cv2.ROTATE_180))
        H2, k1r, k2r = _roma_H(tf.name, r_path, model, device)
        if _reproj_err(k1r, k2r, H2) < err:
            return H2
    return H


def _query_to_global(q_img, r_img, H_q2r, r_global_corners):
    rh, rw = r_img.shape[:2]
    r_local = np.array([[0, 0], [rw, 0], [rw, rh], [0, rh]], np.float32)
    H_r2g, _ = cv2.findHomography(r_local, r_global_corners, cv2.RANSAC)

    qh, qw = q_img.shape[:2]
    q_local = np.array([[0, 0], [qw, 0], [qw, qh], [0, qh]], np.float32)
    q_in_ref = cv2.perspectiveTransform(q_local.reshape(-1, 1, 2), H_q2r).reshape(-1, 2)
    q_global = cv2.perspectiveTransform(q_in_ref.reshape(-1, 1, 2), H_r2g).reshape(-1, 2)
    return q_global.astype(np.float32)


# --------------------------------------------------------------------- #
# main api
# --------------------------------------------------------------------- #
def refine_corners_with_roma(
    query_dir: str | Path,
    ref_dir: str | Path,
    pre_corners: Mapping[int, np.ndarray] | str | Path,
    out_dir: str | Path,
    *,
    reproj_thresh: float = 5.0,
    device: str = "auto",
    logger: logging.Logger | None = None,
) -> dict[str, np.ndarray]:
    log = logger or logging.getLogger(__name__)

    # choose device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_device = torch.device(device)
    log.info("RoMa on %s", torch_device)

    query_dir, ref_dir, out_dir = map(Path, (query_dir, ref_dir, out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    txt_dir: Path | None = None
    if not isinstance(pre_corners, Mapping):
        txt_dir = Path(pre_corners)

    model = roma_outdoor(device=torch_device)

    query_paths = [
        p for p in query_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]
    result: dict[str, np.ndarray] = {}

    for q_path in tqdm(query_paths, desc="RoMa refine"):
        stem = q_path.stem
        ref_path = ref_dir / f"{stem}.JPG"
        if not ref_path.exists():
            ref_path = ref_dir / f"{stem}.jpg"
        if not ref_path.exists():
            log.warning("Ref not found for %s", stem)
            continue

        # --- get ref corners
        if isinstance(pre_corners, Mapping):
            num = int(re.search(r"(\d{3,})", stem).group(1))
            ref_corners = pre_corners.get(num)
        else:
            txt = txt_dir / f"prediction_{stem}.JPG.txt"
            if not txt.exists():
                txt = txt_dir / f"prediction_{stem}.jpg.txt"
            ref_corners = np.loadtxt(txt, np.float32).reshape(4, 2) if txt.exists() else None

        if ref_corners is None or ref_corners.shape != (4, 2):
            log.warning("Corners missing for %s", stem)
            continue

        # --- H estimation
        H_q2r = _best_homography(
            str(q_path), str(ref_path), model, reproj_thresh, torch_device
        )
        if H_q2r is None:
            log.warning("Homography fail %s", stem)
            continue

        # --- project
        q_img = cv2.imread(str(q_path))
        r_img = cv2.imread(str(ref_path))
        q_global = _query_to_global(q_img, r_img, H_q2r, ref_corners)

        np.savetxt(out_dir / f"{stem}_global.txt", q_global, fmt="%.2f")
        result[stem] = q_global

    log.info("Refine done: %d / %d images", len(result), len(query_paths))
    return result

