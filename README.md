# NGC-Geoloc



## 📋 Table of Contents

- [Overview](#overview)
- [Environment Setup](#environment-setup)
- [Requirements](#requirements)
- [Dataset Structure](#dataset-structure)
- [Training Data Generation](#training-data-generation)
- [Model Training](#model-training)
- [Inference](#inference)
- [Evaluation](#evaluation)
- [Reproducing Table II](#reproducing-table-ii)
- [Configuration Files](#configuration-files)
- [Model Weights](#model-weights)
- [Code Structure](#code-structure)

## Overview

This codebase implements a UAV visual localization pipeline that:
1. Generates pseudo ground truth labels using GPS priors and RoMa matching
2. Trains a scene coordinate regression network using pseudo ground truth labels
3. Generates initial corner predictions from network outputs
4. Refines predictions using RoMa (Robust Matching) feature matching
5. Evaluates performance using four metrics: mIoU, center error, yaw error, and success rate

## Environment Setup

### Conda Environment

We provide a conda environment file for easy setup:

```bash
conda env create -f environment.yaml
conda activate ngcgeoloc
```

The `environment.yaml` file contains all required dependencies including:
- PyTorch 2.0.0+ with CUDA support
- OpenCV, NumPy, SciPy
- Transformers (for DINOv2 backbone)
- RoMa (romatch) for feature matching
- Other scientific computing libraries

### Manual Installation

If you prefer manual installation, the key dependencies are:

```bash
pip install torch torchvision
pip install numpy opencv-python
pip install scipy scikit-image
pip install pandas matplotlib tqdm
pip install pyyaml
pip install transformers  # For DINOv2 backbone
pip install romatch  # For RoMa refinement
```

### Hardware Requirements

- **GPU**: NVIDIA GPU with CUDA support (recommended for training and RoMa matching)
- **RAM**: 16GB+ recommended
- **Storage**: Sufficient space for datasets and model weights

## Dataset Structure

The expected dataset structure is:

```
datasets/
├── {scene_name}/
│   ├── train/
│   │   ├── rgb/          # Training RGB images
│   │   └── init/         # Pseudo ground truth coordinate files (.pt)
│   └── test/
│       └── rgb/          # Test/query images
```

## Training Data Generation

The training data generation consists of two main steps:

### Step 1: GPS-based Crop Generation

**Script**: `pseudo_pipeline/crop_allscene_withhomography.py`

This script generates approximate cropped reference images from the DOM (Digital Orthophoto Map) based on GPS priors for query images. It creates initial reference patches that correspond to the expected location of query images within the map coverage area.

**Usage**:
```bash
python pseudo_pipeline/crop_allscene_withhomography.py
```

**Function**: 
- Takes query images with GPS metadata
- Generates corresponding cropped reference images from the DOM
- Outputs reference images that serve as initial matches for the query images

### Step 2: Pseudo Ground Truth Generation via RoMa Matching

**Script**: `pseudo_pipeline/homography_pt2.py`

This script uses RoMa (Robust Matching) feature matching to generate pseudo ground truth pixel correspondences between query images and reference images. It computes homography transformations and generates transformed pixel grids that serve as training labels.

**Usage**:
```bash
python pseudo_pipeline/homography_pt2.py
```

**Function**:
- Performs RoMa matching between query and reference image pairs
- Estimates homography matrices using RANSAC
- Generates transformed pixel grids saved as `.pt` files
- These pixel grids serve as pseudo ground truth labels for training

**Key Parameters**:
- **Pixel grid size**: `1100 × 1650` (height × width) - default grid dimensions
- **Homography estimation**: USAC_MAGSAC method with 5.0 pixel threshold
- **Output format**: Transformed pixel grids saved as `.pt` files

### Training Data Generation Parameters

#### Affine Transformation Parameters
- **Method**: Similarity transform estimation using RANSAC-based affine initialization followed by non-linear least squares optimization
- **Initialization**: `cv2.estimateAffine2D()` with RANSAC
- **Optimization**: `scipy.optimize.least_squares()` for similarity parameters (scale, rotation, translation)

#### Grid and Stride Parameters
- **Output grid size**: `[56, 84]` (height × width) - consistent across all scenes
- **Pixel step**: `8` pixels - spacing between grid points
- **Output subsample**: `8` - network output resolution ratio (1/8 of input size)

#### Data Augmentation Parameters (Training)
- **Rotation range**: `±30 degrees` (`aug_rotation=30`)
- **Scale range**: `[2/3, 3/2]` (`aug_scale_min=2/3`, `aug_scale_max=3/2`)
- **Contrast jitter**: `0.1` (`aug_contrast=0.1`)
- **Brightness jitter**: `0.1` (`aug_brightness=0.1`)
- **Image height**: `448` pixels (input resolution)

## Model Training

### Training Script: `train_2d_xy.py`

Train a scene coordinate regression network for a specific scene:

```bash
python train_2d_xy.py <scene> <net>
```

**Arguments**:
- `<scene>`: Scene name (e.g., `01`, `02`, `03`, etc.) - must exist in `./datasets/{scene}/train/`
- `<net>`: Output path for the trained network weights (e.g., `pretrained_net/scene01.net`)

**Example**:
```bash
python train_2d_xy.py 01 pretrained_net/scene01.net
```

### Training Parameters

The training script uses the following default parameters (can be modified via command-line arguments):

- **Learning rate**: `0.0001` (default)
- **Iterations**: `10,000,000` (default)
- **Mode**: `1` (RGB + GT initialization)
- **Sparse**: Enabled (uses sparse coordinate initialization)
- **Optimizer**: Adam
- **Loss**: L2 norm between predicted and ground truth coordinates

### Training Configuration per Scene

All scenes use the same training hyperparameters. The network architecture uses:
- **Backbone**: DINOv2 Large (frozen, pre-trained)
- **Feature dimension**: 1024 (DINOv2 Large)
- **Reduced channels**: 512 (normal) or 128 (tiny mode)
- **Output**: 2 channels (x, y coordinates)
- **Output resolution**: 56×84 (for 448×672 input)

## Inference

### Prediction Generation: `save_pt_evaluate.py`

Generate predictions for query images using a trained model:

```bash
python save_pt_evaluate.py
```

**Function**:
- Loads the trained network weights
- Processes test/query images from the dataset
- Generates predicted pixel coordinate matrices
- Saves predictions as `.pt` files ready for downstream evaluation

**Output**:
- Prediction files saved as `prediction_{image_name}.pt`
- Each `.pt` file contains a tensor with predicted (x, y) coordinates

**Note**: Before running inference, ensure you have:
1. Trained model weights (see [Model Weights](#model-weights))
2. Updated paths in `save_pt_evaluate.py` to point to:
   - Your trained model: `pretrained_net/{scene}_*.net`
   - Test dataset: `./datasets/{scene}/test`
   - Output directory: `predictions/{scene}`

## Evaluation

### Evaluation Pipeline: `nophi1_project/main.py`

The evaluation pipeline performs the following steps:

1. **Initial Corner Generation**: Converts network predictions to 4-point corner coordinates
2. **Reference Image Cropping**: Crops reference patches from the DOM (Digital Orthophoto Map)
3. **RoMa Refinement**: Refines corners using robust feature matching
4. **Final Cropping** (optional): Generates final cropped reference images
5. **Evaluation**: Computes four metrics against ground truth

### Running Evaluation

```bash
cd nophi1_project
python main.py --config config/scene01.yaml
```

Replace `scene01.yaml` with the appropriate scene configuration file.

**Example for all scenes**:
```bash
cd nophi1_project

# Evaluate each scene
for scene in 01 02 03 04 05 06 09; do
    python main.py --config config/scene${scene}.yaml
done
```

### Evaluation Metrics

The evaluation computes four metrics:

1. **mIoU (mean Intersection over Union)**: Average IoU between predicted and ground truth polygons
   - Computed for both origin (initial) and optimized (refined) predictions

2. **Center Error (meters)**: Euclidean distance between predicted and ground truth centroids
   - Converted to meters using `pixel_scale` parameter
   - Reported as median error

3. **Yaw Error (degrees)**: Angular difference in orientation
   - Computed using SVD-based principal direction estimation
   - Reported as median error

4. **Success Rate**: Percentage of predictions with IoU > 0.9
   - Only computed for optimized predictions

### Output Files

Results are saved as:
- **CSV files**: `outputs/{scene}/eval/results.csv` - Detailed per-image metrics
- **LaTeX tables**: `outputs/{scene}/eval/results.tex` - LaTeX table format for paper inclusion

The CSV files include:
- Per-image metrics (IoU, center error, yaw error)
- Summary row with scene-level statistics (mean/median values)

## Reproducing Table II

To reproduce the scene-level metrics reported in Table II:

### Step 1: Prepare Data

Ensure you have:
- Trained model weights for each scene (see [Model Weights](#model-weights))
- Test images in the expected directory structure
- Ground truth files in the format: `match_{file_id}_transformed_pixel_grid.pt`
- DOM (Digital Orthophoto Map) images for each scene

### Step 2: Generate Predictions

For each scene, generate predictions:

```bash
python save_pt_evaluate.py
```

Update the paths in `save_pt_evaluate.py` to point to:
- Your trained model: `pretrained_net/{scene}_*.net`
- Test dataset: `./datasets/{scene}/test`
- Output directory: `predictions/{scene}`

### Step 3: Update Configuration Files

Edit the YAML configuration files in `nophi1_project/config/` to match your data paths:

```yaml
device: auto
dom_tif: /path/to/scene{XX}/cropped_satellite{XX}.tif
pt_dir: /path/to/predictions/{scene}
query_dir: /path/to/datasets/{scene}/test/rgb
gt_dir: /path/to/ground_truth/homographies{XX}
pixel_scale: 0.3  # meters per pixel (adjust per scene)
grid_size: [56, 84]
pixel_step: 8
reproj_thresh: 5.0
```

### Step 4: Run Evaluation for All Scenes

```bash
cd nophi1_project

# Evaluate each scene
for scene in 01 02 03 04 05 06 09; do
    python main.py --config config/scene${scene}.yaml
done
```

### Step 5: Collect Results

Results are saved as CSV files:
- `outputs/{scene}/eval/results.csv` - Detailed per-image metrics
- `outputs/{scene}/eval/results.tex` - LaTeX table format

The CSV files include:
- Per-image metrics (IoU, center error, yaw error)
- Summary row with scene-level statistics (mean/median values)

### Step 6: Aggregate Scene-Level Metrics

To create Table II, extract the summary statistics from each scene's `results.csv`:

```python
import pandas as pd

scenes = ['01', '02', '03', '04', '05', '06', '09']
results = []

for scene in scenes:
    df = pd.read_csv(f'outputs/{scene}/eval/results.csv')
    summary = df[df['file_id'] == 'SUMMARY'].iloc[0]
    results.append({
        'Scene': scene,
        'mIoU (Origin)': summary['iou_origin'],
        'mIoU (Optimized)': summary['iou_optimized'],
        'Med. Center Error (m) - Origin': summary['center_error_origin'],
        'Med. Center Error (m) - Optimized': summary['center_error_optimized'],
        'Med. Yaw Error (°) - Origin': summary['yaw_error_origin'],
        'Med. Yaw Error (°) - Optimized': summary['yaw_error_optimized'],
        'Success Rate': summary['success_rate_optimized']
    })

table = pd.DataFrame(results)
print(table.to_string(index=False))
table.to_csv('table_ii_results.csv', index=False)
```

## Configuration Files

Each scene has a dedicated YAML configuration file in `nophi1_project/config/`:

- `scene01.yaml` - Scene 01 configuration
- `scene02.yaml` - Scene 02 configuration
- `scene03.yaml` - Scene 03 configuration
- `scene04.yaml` - Scene 04 configuration
- `scene05.yaml` - Scene 05 configuration
- `scene06.yaml` - Scene 06 configuration
- `scene09.yaml` - Scene 09 configuration

### Configuration Parameters

Each config file specifies:

- **Paths**:
  - `dom_tif`: Path to DOM image
  - `pt_dir`: Directory containing prediction `.pt` files
  - `query_dir`: Directory containing query/test images
  - `gt_dir`: Directory containing ground truth files
  - `eval_csv`: Output path for evaluation CSV
  - `eval_vis_dir`: Output directory for visualizations

- **Processing Parameters**:
  - `grid_size`: `[56, 84]` - Output grid dimensions
  - `pixel_step`: `8` - Grid point spacing
  - `pixel_scale`: `0.3` - Meters per pixel (scene-dependent)
  - `reproj_thresh`: `5.0` - Reprojection error threshold for RoMa

- **Output Directories**:
  - `pre_corners_dir`: Initial corner predictions
  - `ref_img_dir`: Cropped reference images
  - `global_corner_dir`: Refined corner predictions
  - `final_crop_dir`: Final cropped images (optional)

## Model Weights

### Pre-trained Models

Due to the large file size of model weights, pre-trained models are provided separately via **Baidu Netdisk (百度网盘)**. 

**Download Link**: [Will be provided upon acceptance]

The expected naming convention for model weights is:
```
pretrained_net/
├── {scene}_*.net  # Trained weights for each scene
```

### Model Architecture

- **Backbone**: DINOv2 Large (frozen, pre-trained)
- **Feature dimension**: 1024 (DINOv2 Large)
- **Reduced channels**: 512 (normal) or 128 (tiny)
- **Output**: 2 channels (x, y coordinates)
- **Output resolution**: 56×84 (for 448×672 input)

### Loading Models

Models are loaded using:

```python
from network import Network
import torch

mean = torch.zeros(2)  # Scene-specific mean (computed from training data)
network = Network(mean, tiny=False).cuda()
network.load_state_dict(torch.load('pretrained_net/{scene}_*.net'))
```

## Code Structure

```
NGC-Geoloc/
├── environment.yaml              # Conda environment file
├── train_2d_xy.py                # Training script
├── save_pt_evaluate.py           # Inference/prediction generation
├── dataset.py                    # Dataset loader with augmentation
├── network.py                    # Network architecture (DINOv2 + decoder)
├── nophi1_project/
│   ├── main.py                   # Main evaluation pipeline
│   └── config/                   # Scene configuration files
│       ├── scene01.yaml
│       ├── scene02.yaml
│       ├── scene03.yaml
│       ├── scene04.yaml
│       ├── scene05.yaml
│       ├── scene06.yaml
│       └── scene09.yaml
├── pipelines_nophi1/
│   ├── coordinate.py             # Corner generation (similarity transform)
│   ├── cropping.py               # Reference image cropping
│   ├── refinement.py             # RoMa-based refinement
│   └── evaluation.py             # Metric computation
└── pseudo_pipeline/
    ├── crop_allscene_withhomography.py  # GPS-based crop generation
    └── homography_pt2.py               # Pseudo GT generation via RoMa
```

## Workflow Summary

The complete workflow is:

1. **Training Data Generation**:
   ```bash
   # Step 1: Generate reference crops from GPS priors
   python pseudo_pipeline/crop_allscene_withhomography.py
   
   # Step 2: Generate pseudo GT via RoMa matching
   python pseudo_pipeline/homography_pt2.py
   ```

2. **Model Training**:
   ```bash
   python train_2d_xy.py <scene> <net>
   ```

3. **Inference**:
   ```bash
   python save_pt_evaluate.py
   ```

4. **Evaluation**:
   ```bash
   cd nophi1_project
   python main.py --config config/scene{XX}.yaml
   ```

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce batch size or use CPU mode (`device: cpu` in config)
2. **Missing ground truth files**: Ensure GT files follow naming: `match_{file_id}_transformed_pixel_grid.pt`
3. **Path errors**: Verify all paths in YAML config files are correct and accessible
4. **RoMa import error**: Install romatch: `pip install romatch`
5. **DINOv2 model not found**: Ensure DINOv2 model is downloaded to `./dinov2_large/` directory

### Verification Steps

1. Check that prediction files exist: `ls {pt_dir}/*.pt`
2. Verify ground truth directory: `ls {gt_dir}/match_*.pt`
3. Check DOM image is readable: `python -c "import cv2; print(cv2.imread('{dom_tif}') is not None)"`
4. Verify conda environment: `conda env list` (should show environment)





---

**Note**: This reproducibility package is released upon acceptance of the manuscript. The public URL and Baidu Netdisk link for model weights will be provided in the camera-ready version.
