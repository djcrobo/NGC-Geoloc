#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""UAV-VisLoc pipeline runner (no CSV phi1 dependency)"""

import argparse, logging, yaml, sys
from pathlib import Path
from datetime import datetime

# Ensure repo root in path
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from pipelines_nophi1.coordinate import gen_initial_corners_nophi1
from pipelines_nophi1.cropping import crop_dom_by_corners
from pipelines_nophi1.refinement import refine_corners_with_roma
from pipelines_nophi1.evaluation import evaluate_corners


def setup_logger(level: str = "INFO"):
    fmt = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
    logging.basicConfig(format=fmt, level=getattr(logging, level.upper(), 20))
    return logging.getLogger("PIPELINE-NOPHI1")


def parse_cfg():
    p = argparse.ArgumentParser(description="UAV-VisLoc full pipeline (no phi1 csv)")
    p.add_argument("-c", "--config", default=str(ROOT/"nophi1_project/config/scene01.yaml"), help="YAML config path")
    args = p.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg_dict = yaml.safe_load(f)
    cfg_dict['cfg_path'] = args.config
    return argparse.Namespace(**cfg_dict)


def main():
    log = setup_logger()
    cfg = parse_cfg()
    log.info("Start pipeline (no-phi1) | cfg=%s", cfg.cfg_path)

    t0 = datetime.now()

    # Step 1: initial corners (computed without csv phi1)
    init_corners = gen_initial_corners_nophi1(
        pt_dir     = cfg.pt_dir,
        out_dir    = cfg.pre_corners_dir,
        grid_size  = tuple(cfg.grid_size),
        pixel_step = cfg.pixel_step,
    )

    # Step 2: crop reference images
    crop_dom_by_corners(
        dom_path = cfg.dom_tif,
        corners  = init_corners,
        out_dir  = cfg.ref_img_dir,
    )

    # Step 3: RoMa refinement
    opt_corners = refine_corners_with_roma(
        query_dir     = cfg.query_dir,
        ref_dir       = cfg.ref_img_dir,
        pre_corners   = init_corners,
        out_dir       = cfg.global_corner_dir,
        reproj_thresh = cfg.reproj_thresh,
        device        = cfg.device,
    )

    # Step 4: optional final crop
    if getattr(cfg, 'final_crop_dir', None):
        crop_dom_by_corners(
            dom_path = cfg.dom_tif,
            corners  = opt_corners,
            out_dir  = cfg.final_crop_dir,
        )

    # Step 5: evaluation
    evaluate_corners(
        origin_corners_dir = cfg.pre_corners_dir,
        optimized_dir      = cfg.global_corner_dir,
        gt_dir             = cfg.gt_dir,
        dom_img            = cfg.dom_tif,
        out_csv            = cfg.eval_csv,
        vis_dir            = cfg.eval_vis_dir,
        pixel_scale        = cfg.pixel_scale,
        show_text_overlay  = cfg.show_text,
    )

    log.info("Pipeline finished in %s", datetime.now() - t0)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.exception("Pipeline aborted: %s", e)
        sys.exit(1)


