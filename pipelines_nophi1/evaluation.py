
"""
pipelines_nophi1/evaluation.py  ——  
"""

from __future__ import annotations
import logging, re, textwrap
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import cv2
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm


# -------------------------------------------------- 原始 / 优化角点解析
_NUM = re.compile(r'^\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)\s+'
                  r'([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)\s*$')
_XY  = re.compile(r'x=([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)'
                  r'.*?y=([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)')

def _origin(fp: str | Path) -> np.ndarray:
    pts: List[List[float]] = []
    for ln in Path(fp).read_text(encoding="utf-8", errors="ignore").splitlines():
        ln = ln.strip()
        m = _NUM.match(ln) or _XY.search(ln)
        if m:
            pts.append([float(m.group(1)), float(m.group(2))])
    return np.asarray(pts, np.float32)

def _opt(fp: str | Path) -> np.ndarray:
    return np.loadtxt(fp, np.float32).reshape(-1, 2)


# -------------------------------------------------- GT：凸包
def _gt(fp: str | Path) -> np.ndarray:
    g = torch.load(fp, map_location="cpu")
    pts = (g.cpu().numpy() if torch.is_tensor(g) else g) \
          .transpose(1, 2, 0).reshape(-1, 2).astype(np.float32)
    return cv2.convexHull(pts).squeeze()


# -------------------------------------------------- 几何工具
def _centroid(poly: np.ndarray) -> np.ndarray:
    M = cv2.moments(poly.reshape(-1, 1, 2))
    if abs(M["m00"]) < 1e-6:
        return poly.mean(0)
    return np.array([M["m10"] / M["m00"], M["m01"] / M["m00"]], np.float32)

def _mask(poly: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    m = np.zeros(shape, np.uint8)
    cv2.fillPoly(m, [poly.astype(np.int32)], 255)
    return m

def _yaw(poly: np.ndarray) -> float:
    pts = poly.astype(np.float32) - poly.mean(0)
    if pts.shape[0] < 2:
        return 0.0
    _, _, vt = np.linalg.svd(pts, full_matrices=False)
    v = vt[0]
    return (np.degrees(np.arctan2(v[1], v[0])) + 180) % 180

def _yaw_err(a: float, b: float) -> float:
    d = abs(a - b) % 180
    return d if d <= 90 else 180 - d


# -------------------------------------------------- 指标
def _metrics(o, p, g, scale: float) -> dict:
    H, W = np.ceil(np.vstack([o, p, g]).max(0)).astype(int) + 100
    iou = lambda A, B: np.logical_and(A, B).sum() / (np.logical_or(A, B).sum() + 1e-6)
    mo, mp, mg = (_mask(x, (H, W)) for x in (o, p, g))
    co, cp, cg = (_centroid(x) for x in (o, p, g))
    err_o, err_p = np.linalg.norm(co - cg) * scale, np.linalg.norm(cp - cg) * scale
    yo, yp, yg = (_yaw(x) for x in (o, p, g))
    yerr_o, yerr_p = _yaw_err(yo, yg), _yaw_err(yp, yg)

    return dict(
        iou_origin=iou(mo, mg),
        iou_optimized=iou(mp, mg),
        center_error_origin=err_o,
        center_error_optimized=err_p,
        error_improvement=err_o - err_p,
        yaw_error_origin=yerr_o,
        yaw_error_optimized=yerr_p,
        yaw_error_improvement=yerr_o - yerr_p,
    )


# -------------------------------------------------- 三线表（只统计 Optimized 成功率）
def _latex_table(miou_o, miou_p,
                 med_c_o, med_c_p,
                 med_y_o, med_y_p,
                 succ_p, total) -> str:
    return textwrap.dedent(f"""\
        \\begin{{tabular}}{{lcc}}
        \\toprule
        指标 & Origin & Optimized \\\\
        \\midrule
        mIoU (mean) & {miou_o:.3f} & {miou_p:.3f} \\\\
        Median center error (m) & {med_c_o:.2f} & {med_c_p:.2f} \\\\
        Median yaw error ($^\\circ$) & {med_y_o:.2f} & {med_y_p:.2f} \\\\
        Success rate (IoU$>$0.9) & -- & {succ_p}/{total} \\\\
        \\bottomrule
        \\end{{tabular}}""")


# -------------------------------------------------- 主函数
def evaluate_corners(
    origin_corners_dir: str | Path,
    optimized_dir: str | Path,
    gt_dir: str | Path,
    dom_img: str | Path,
    out_csv: str | Path,
    vis_dir: str | Path,
    *,
    pixel_scale: float = .3,
    show_text_overlay: bool = True,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    log = logger or logging.getLogger(__name__)
    origin_corners_dir, optimized_dir, gt_dir = map(Path, (origin_corners_dir, optimized_dir, gt_dir))
    vis_dir, out_csv = map(Path, (vis_dir, out_csv))
    vis_dir.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    dom = cv2.cvtColor(cv2.imread(str(dom_img)), cv2.COLOR_BGR2RGB)

    res, skip = [], []
    for txt in tqdm(sorted(origin_corners_dir.glob("*.txt")), desc="Eval"):
        nums = re.findall(r"(\d{3,})", txt.name)
        if not nums:
            continue
        fid = nums[-1].lstrip("0")

        o = _origin(txt)
        if o.shape != (4, 2):
            skip.append(f"{txt.name} origin malformed")
            continue

        stem = txt.stem.replace("prediction_", "").split(".")[0]
        p_path = optimized_dir / f"{stem}_global.txt"
        g_path = gt_dir / f"match_{fid}_transformed_pixel_grid.pt"
        if not p_path.exists():
            skip.append(f"{stem} no opt")
            continue
        if not g_path.exists():
            skip.append(f"{stem} no GT")
            continue

        p, g = _opt(p_path), _gt(g_path)
        if p.shape[0] < 4 or g.shape[0] < 3:
            skip.append(f"{stem} insufficient pts")
            continue

        res.append(dict(file_id=fid, **_metrics(o, p, g, pixel_scale)))

        # 可视化（如需）：
        # _draw(fid, o, p, g, dom, vis_dir / f"{fid}.png", res[-1], show_text_overlay)

    df = pd.DataFrame(res)
    if not df.empty:
        total = len(df)
        miou_o, miou_p = df.iou_origin.mean(), df.iou_optimized.mean()
        med_c_o, med_c_p = df.center_error_origin.median(), df.center_error_optimized.median()
        med_y_o, med_y_p = df.yaw_error_origin.median(), df.yaw_error_optimized.median()
        succ_p = (df.iou_optimized > .9).sum()

        summary = dict(file_id="SUMMARY",
                       iou_origin=miou_o, iou_optimized=miou_p,
                       center_error_origin=med_c_o, center_error_optimized=med_c_p,
                       error_improvement=med_c_o - med_c_p,
                       yaw_error_origin=med_y_o, yaw_error_optimized=med_y_p,
                       yaw_error_improvement=med_y_o - med_y_p,
                       success_rate_optimized=f"{succ_p}/{total}")

        df = pd.concat([df, pd.DataFrame([summary])], ignore_index=True)
        df.to_csv(out_csv, index=False)
        log.info("CSV → %s", out_csv.resolve())

        # LaTeX 三线表
        tex = _latex_table(miou_o, miou_p,
                           med_c_o, med_c_p,
                           med_y_o, med_y_p,
                           succ_p, total)
        Path(out_csv).with_suffix(".tex").write_text(tex, encoding="utf-8")
        log.info("\n%s", tex)

    if skip:
        log.warning("Skipped %d files:\n%s",
                    len(skip), "\n".join(skip[:15] + ["..."] if len(skip) > 15 else skip))
    return df

