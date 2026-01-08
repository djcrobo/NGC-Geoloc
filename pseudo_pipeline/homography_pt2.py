import os
import cv2
from PIL import Image
import numpy as np
import torch
from romatch import roma_outdoor  # Ensure romatch is installed and imported correctly
import random
import re
import matplotlib.pyplot as plt
import gc

# Select CUDA or MPS device if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.backends.mps.is_available():
    device = torch.device('mps')


def extract_numbers_query(filename):
    """
    Extract the last three consecutive digits from a query filename and convert to integer.
    Example: '06_0073.JPG' -> 73
    """
    match = re.search(r'(\d{3})\D*$', filename)
    if match:
        return int(match.group(1))
    return None


def extract_numbers_ref(filename):
    """
    Extract all digit sequences from a reference filename, concatenate them, and convert to integer.
    Example: 'ref_1234_image.JPG' -> 1234
    """
    matches = re.findall(r'\d+', filename)
    if matches:
        return int(''.join(matches))
    return None


def translatePoint(p, d):
    x, y = p
    dx, dy = d
    return [x + dx, y + dy]


def homographyTransform(h, p):
    res = h @ p
    res = res / res[-1]
    return res[:2].astype('int')


def draw_keypoint_matches(im1_path, im2_path, kpts1, kpts2, save_prefix):
    """
    Draw and save keypoint matching visualization.
    """
    img1 = cv2.imread(im1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(im2_path, cv2.IMREAD_GRAYSCALE)

    keypoints1 = [cv2.KeyPoint(x[0], x[1], 1) for x in kpts1]
    keypoints2 = [cv2.KeyPoint(x[0], x[1], 1) for x in kpts2]

    matches = [cv2.DMatch(i, i, 0) for i in range(len(keypoints1))]

    match_img = cv2.drawMatches(
        img1, keypoints1,
        img2, keypoints2,
        matches, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    cv2.imwrite(f'{save_prefix}_matches.jpg', match_img)


def estimate_3d_homography(correspondences):
    """
    Estimate a 3D homography matrix (used for affine transformations).
    """
    a_list = []
    for corr in correspondences:
        (x1, y1, z1), (x2, y2, z2) = corr
        xp2 = x2 / z2
        yp2 = y2 / z2
        ax = [-x1, -y1, -1, 0, 0, 0, xp2 * x1, xp2 * y1, xp2]
        ay = [0, 0, 0, -x1, -y1, -1, yp2 * x1, yp2 * y1, yp2]
        a_list.append(ax)
        a_list.append(ay)
    mat_a = np.array(a_list)
    _, _, v = np.linalg.svd(mat_a)
    h = np.reshape(v[-1], (3, 3))
    return h / h[-1, -1]


def estimate_2d_homography(correspondences):
    """
    Estimate a 2D homography matrix.
    """
    new_correspondences = [
        ((x1, y1, 1), (x2, y2, 1))
        for ((x1, y1), (x2, y2)) in correspondences
    ]
    return estimate_3d_homography(new_correspondences)


def geometric_distance(correspondence, h):
    """
    Compute geometric reprojection error for RANSAC inlier checking.
    """
    (x1, y1), (x2, y2) = correspondence
    X = np.array([x1, y1, 1])
    predicted_x2, predicted_y2 = homographyTransform(h, X)[:2]
    distance = np.sqrt((x2 - predicted_x2) ** 2 + (y2 - predicted_y2) ** 2)
    return distance


def fsc(small, big, threshold=1, N_iteration=2000):
    """
    RANSAC-based homography estimation.
    """
    n_max_inliers = 0
    best_h = None
    for _ in range(N_iteration):
        samples = random.sample(small, 3)
        estimated_h = estimate_2d_homography(samples)
        best_correspondence = []
        for corres in big:
            dist = geometric_distance(corres, estimated_h)
            if dist < threshold:
                best_correspondence.append(corres)
        if len(best_correspondence) > n_max_inliers:
            best_h = estimate_2d_homography(best_correspondence)
            n_max_inliers = len(best_correspondence)
    mask = [geometric_distance(corr, best_h) < threshold for corr in big]
    return best_h, mask


def transform_image(image_path, H, target_size):
    """
    Warp an image using homography matrix H.
    """
    img = cv2.imread(image_path)
    transformed_img = cv2.warpPerspective(img, H, target_size)
    return transformed_img


def overlay_images(image1, image2, alpha=0.5):
    """
    Overlay two images with alpha blending.
    """
    if image1.shape != image2.shape:
        raise ValueError("Images must have the same dimensions to be overlaid.")
    blended_img = cv2.addWeighted(image1, alpha, image2, 1 - alpha, 0)
    return blended_img


def split_and_merge(image1, image2, num_blocks=2):
    """
    Split images into blocks and merge them alternately (checkerboard visualization).
    """
    height, width = image1.shape[:2]
    block_width = width // num_blocks
    block_height = height // num_blocks
    result_image = Image.new('RGB', (width, height))
    for row in range(num_blocks):
        for col in range(num_blocks):
            if (row + col) % 2 == 0:
                block_img = image1[
                    row * block_height:(row + 1) * block_height,
                    col * block_width:(col + 1) * block_width
                ]
            else:
                block_img = image2[
                    row * block_height:(row + 1) * block_height,
                    col * block_width:(col + 1) * block_width
                ]
            result_image.paste(
                Image.fromarray(block_img),
                (col * block_width, row * block_height)
            )
    return result_image


def generate_pixel_grid(height=1100, width=1650):
    """
    Generate a pixel grid tensor of shape (2, height, width).
    Top-left corner is (0, 0), x increases horizontally, y increases vertically.
    """
    pixel_grid = torch.zeros((2, height, width), dtype=torch.float32)
    for y in range(height):
        for x in range(width):
            pixel_grid[0, y, x] = x
            pixel_grid[1, y, x] = y
    return pixel_grid


def apply_homography_to_grid(pixel_grid, H_new):
    """
    Apply homography to a pixel grid.

    Args:
        pixel_grid (torch.Tensor): shape (2, H, W)
        H_new (torch.Tensor or np.ndarray): 3x3 homography matrix

    Returns:
        torch.Tensor: transformed pixel grid
    """
    if isinstance(H_new, torch.Tensor):
        H_new = H_new.numpy()

    _, H, W = pixel_grid.shape

    ones = torch.ones((1, H, W))
    pixel_grid_homogeneous = torch.cat([pixel_grid, ones], dim=0)
    pixel_grid_homogeneous = pixel_grid_homogeneous.view(3, -1)

    transformed = H_new @ pixel_grid_homogeneous.numpy()
    transformed /= transformed[2, :]
    transformed = transformed[:2, :].reshape(2, H, W)

    return torch.from_numpy(transformed).float()


def reverse_rotation_in_homography(H):
    """
    Reverse the rotation component of a homography matrix.
    """
    R = H[:2, :2]
    t = H[:2, 2]

    R_reversed = R.T

    H_reversed = np.eye(3)
    H_reversed[:2, :2] = R_reversed
    H_reversed[:2, 2] = -R_reversed @ t

    return H_reversed
