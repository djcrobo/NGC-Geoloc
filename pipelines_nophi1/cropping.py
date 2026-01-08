from __future__ import annotations
import logging, re
from pathlib import Path
from typing import Mapping
import numpy as np
import cv2


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def _read_txt(txt: Path) -> np.ndarray:
    """
    Read 4 corner coordinates from a text file.
    Supports both 'x=..., y=...' and plain numeric formats.
    """
    pts = []
    for ln in txt.read_text().splitlines():
        ln = ln.strip()
        if not ln:
            continue

        m = re.match(r"^([\d.]+)\s+([\d.]+)$", ln)
        if m:
            pts.append([float(m.group(1)), float(m.group(2))])
            continue

        if "x=" in ln and "y=" in ln:
            try:
                x = float(ln.split("x=")[1].split(",")[0])
                y = float(ln.split("y=")[1].split(")")[0])
                pts.append([x, y])
            except Exception:
                pass
    return np.asarray(pts, np.float32)


def _id_from_stem(stem: str) -> str:
    """Extract ID pattern like 02_0533 from filename stem."""
    m = re.search(r"(\d+_\d+)", stem)
    if not m:
        raise ValueError(f"Failed to extract ID from {stem}")
    return m.group(1)


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------
def crop_dom_by_corners(
    dom_path: str | Path,
    out_dir: str | Path,
    *,
    corners: Mapping[int, np.ndarray] | None = None,
    corners_txt: str | Path | None = None,
    logger: logging.Logger | None = None,
) -> None:
    """
    Crop patches from a georeferenced basemap using 4-point corner coordinates.

    Parameters
    ----------
    dom_path   : str | Path
        Input basemap image (tif/png/jpg)
    out_dir    : str | Path
        Output directory for cropped patches
    corners    : dict[num or stem → (4,2)], optional
        Corner coordinates in pixel space
    corners_txt: Path, optional
        Directory containing *.txt corner files (used if corners=None)
    """
    log = logger or logging.getLogger(__name__)
    dom_path, out_dir = Path(dom_path), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------------- 1. Infer scene prefix ----------------
    scene_prefix: str | None = None
    m = re.search(r"[\\/](\d{2})(?:[\\/]|$)", str(dom_path.parent))
    if not m:
        m = re.search(r"(\d{2})\D*\.tif$", dom_path.name, re.IGNORECASE)
    if m:
        scene_prefix = m.group(1)
        log.debug("Scene prefix inferred: %s", scene_prefix)

    # ---------------- 2. Load DOM image ---------------------
    img = cv2.imread(str(dom_path))
    if img is None:
        raise FileNotFoundError(f"Cannot read DOM image: {dom_path}")

    # ---------------- 3. Build cropping tasks ---------------
    items: list[tuple[str, np.ndarray]] = []

    if corners is not None:
        # Using dictionary input
        for k, v in corners.items():
            if not isinstance(v, np.ndarray) or v.shape != (4, 2):
                continue

            if isinstance(k, str) and "_" in k:
                stem = k
            else:
                num = int(k)
                if scene_prefix:
                    stem = f"{scene_prefix}_{num:04d}"
                else:
                    stem = f"{num:02d}_{num:04d}"
            items.append((stem, v))
    else:
        # Read from txt directory
        txt_dir = Path(corners_txt)
        for txt in txt_dir.glob("*.txt"):
            pts = _read_txt(txt)
            if pts.shape != (4, 2):
                log.warning("Skipping invalid corner file %s", txt.name)
                continue
            items.append((_id_from_stem(txt.stem), pts))

    if not items:
        log.warning("No valid corner definitions found.")
        return

    # ---------------- 4. Perform cropping -------------------
    for stem, pts in items:
        width = int(
            max(np.linalg.norm(pts[0] - pts[1]), np.linalg.norm(pts[2] - pts[3]))
        )
        height = int(
            max(np.linalg.norm(pts[0] - pts[3]), np.linalg.norm(pts[1] - pts[2]))
        )

        dst = np.array(
            [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
            np.float32,
        )
        M = cv2.getPerspectiveTransform(pts, dst)
        warped = cv2.warpPerspective(img, M, (width, height))
        out_file = out_dir / f"{stem}.jpg"
        cv2.imwrite(str(out_file), warped)
        log.debug("Cropped %s → %s", stem, out_file.name)

