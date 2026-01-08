import torch
import torch.nn as nn
import os
import torch.nn.functional as F
import numpy as np
from dataset import CamLocDataset
from network import Network
import time

# Example dataset (update with your own paths)

testset = CamLocDataset("./datasets/try3_test/test", mode=1, sparse=1, augment=False)

# DataLoader
test_loader = torch.utils.data.DataLoader(testset, shuffle=False, num_workers=6)

# Compute mean scene coordinate
mean = torch.zeros((2))
count = 0
for image, gt_pose, gt_coords, focal_length, file in test_loader:
    gt_coords = gt_coords[0].view(2, -1)
    coord_mask = gt_coords.abs().sum(0) > 0
    if coord_mask.sum() > 0:
        gt_coords = gt_coords[:, coord_mask]
        mean += gt_coords.sum(1)
        count += coord_mask.sum()
mean /= count

# Initialize and load model
network = Network(mean, tiny=False).cuda()
model_path = "pretrained_net/03_618new.net"
network.load_state_dict(torch.load(model_path))

def evaluate_and_save_processed_prediction(network, test_loader, save_dir="predictions_eval", num_samples=100):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    total_samples = 0
    start_time = time.time()

    with torch.no_grad():
        for idx, (image, gt_pose, gt_coords, focal_length, file) in enumerate(test_loader):
            if total_samples >= num_samples:
                break

            image = image.cuda()
            gt_offsets = gt_coords.cuda()

            # Forward pass
            pred_offsets = network(image)
            scene_coords = pred_offsets[0].view(2, -1)
            gt_coords = gt_offsets[0].view(2, -1).cuda()
            gt_coords_mask = gt_coords.abs().sum(0) > 0
            loss = torch.norm(scene_coords - gt_coords, dim=0, p=2)[gt_coords_mask].mean()

            # Scale to pixel coordinates
            width_multiplier = 3500
            height_multiplier = 3500
            pred_offsets[:, 0, :, :] *= width_multiplier
            pred_offsets[:, 1, :, :] *= height_multiplier
            gt_offsets[:, 0, :, :] *= width_multiplier
            gt_offsets[:, 1, :, :] *= height_multiplier

            # Compute mean per-pixel error
            pred_x, pred_y = pred_offsets[0, 0], pred_offsets[0, 1]
            gt_x, gt_y = gt_offsets[0, 0], gt_offsets[0, 1]
            distance_diff = torch.sqrt((pred_x - gt_x) ** 2 + (pred_y - gt_y) ** 2)
            avg_diff = distance_diff.mean().item()

            print(f"[{idx:04d}] {os.path.basename(file[0])} | Mean pixel error: {avg_diff:.2f}")

            # Save prediction
            prediction_filename = f"prediction_{os.path.basename(file[0])}.pt"
            prediction_path = os.path.join(save_dir, prediction_filename)
            torch.save(pred_offsets, prediction_path)
            print(f"Saved: {prediction_path}")

            total_samples += 1

    elapsed = time.time() - start_time
    print(f"\nCompleted {total_samples} samples in {elapsed:.2f}s "
          f"({elapsed / max(total_samples, 1):.2f}s per image).")

    return 0

# Run evaluation
evaluate_and_save_processed_prediction(network, test_loader, num_samples=4500)
