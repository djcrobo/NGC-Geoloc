import torch
import torch.optim as optim
import time
import argparse
import math
from dataset import CamLocDataset
from network import Network
import matplotlib.pyplot as plt
from tqdm import tqdm

# -------------------------------------------------------
# Argument parser
# -------------------------------------------------------
parser = argparse.ArgumentParser(
    description='Initialize a scene coordinate regression network.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('scene', help='name of a scene in the dataset folder')
parser.add_argument('network', help='output file name for the network')

parser.add_argument('--learningrate', '-lr', type=float, default=0.0001, 
    help='learning rate')

parser.add_argument('--iterations', '-iter', type=int, default=10000000,
    help='number of training iterations (model updates)')

parser.add_argument('--inittolerance', '-itol', type=float, default=0.1, 
    help='switch to reprojection optimization when prediction is within this tolerance (m)')

parser.add_argument('--mindepth', '-mind', type=float, default=0.1, 
    help='enforce predicted scene coordinates to be at least this far in front of camera (m)')

parser.add_argument('--maxdepth', '-maxd', type=float, default=1000, 
    help='enforce predicted scene coordinates to be at most this far in front of camera (m)')

parser.add_argument('--targetdepth', '-td', type=float, default=10, 
    help='proxy depth if ground truth is unavailable (m)')

parser.add_argument('--softclamp', '-sc', type=float, default=100, 
    help='square root loss after this threshold (px)')

parser.add_argument('--hardclamp', '-hc', type=float, default=1000, 
    help='clamp loss with this threshold (px)')

parser.add_argument('--mode', '-m', type=int, default=1, choices=range(3),
    help='training mode: 0 = RGB only, 1 = RGB + GT, 2 = RGB-D')

parser.add_argument('--sparse', '-sparse', action='store_true',
    help='for mode 1, use sparse coordinate initialization targets')

parser.add_argument('--tiny', '-tiny', action='store_true',
    help='train a small-capacity model')

parser.add_argument('--session', '-sid', default='',
    help='custom session name for output separation')

opt = parser.parse_args()

# -------------------------------------------------------
# Dataset and initialization
# -------------------------------------------------------
use_init = opt.mode > 0
opt.sparse = 1
trainset = CamLocDataset("./datasets/" + opt.scene + "/train",
                         mode=min(opt.mode, 1),
                         sparse=opt.sparse,
                         augment=False)
trainset_loader = torch.utils.data.DataLoader(trainset, shuffle=True, num_workers=0)

print("Found %d training images for %s." % (len(trainset), opt.scene))
print("Calculating mean scene coordinate...")

loss_values = []
mean = torch.zeros((2))
count = 0

for image, gt_pose, gt_coords, mask, file in trainset_loader:
    if use_init:
        gt_coords = gt_coords[0].view(2, -1)
        coord_mask = gt_coords.abs().sum(0) > 0
        if coord_mask.sum() > 0:
            gt_coords = gt_coords[:, coord_mask]
            mean += gt_coords.sum(1)
            count += coord_mask.sum()
    else:
        mean += gt_pose[0, 0:3, 3]
        count += 1
mean /= count
print("Done. Mean: %.2f,  %.2f\n" % (mean[0], mean[1]))

# -------------------------------------------------------
# Network and optimizer
# -------------------------------------------------------
network = Network(mean, opt.tiny).cuda()
network.train()
optimizer = optim.Adam(network.parameters(), lr=opt.learningrate)
iteration = 0
epochs = int(opt.iterations / len(trainset))

log_filename = 'log_init_%s_%s.txt' % (opt.scene, opt.session)
train_log = open(log_filename, 'w')

# -------------------------------------------------------
# Pixel grid for reprojection computation
# -------------------------------------------------------
pixel_grid = torch.zeros((2,
    math.ceil(5000 / network.OUTPUT_SUBSAMPLE),
    math.ceil(5000 / network.OUTPUT_SUBSAMPLE)))

for x in range(pixel_grid.size(2)):
    for y in range(pixel_grid.size(1)):
        pixel_grid[0, y, x] = x * network.OUTPUT_SUBSAMPLE + network.OUTPUT_SUBSAMPLE / 2
        pixel_grid[1, y, x] = y * network.OUTPUT_SUBSAMPLE + network.OUTPUT_SUBSAMPLE / 2
pixel_grid = pixel_grid.cuda()

# -------------------------------------------------------
# Training loop
# -------------------------------------------------------
for epoch in range(epochs):
    print("=== Epoch: %d ======================================" % epoch)
    epoch_loss = 0.0
    num_batches = 0

    for image, gt_pose, gt_coords, mask, file in trainset_loader:
        start_time = time.time()
        num_batches += 1

        # Preprocess input
        image = image * mask.clone()
        scene_coords = network(image.cuda())

        # Compute loss
        if opt.mode == 1:
            scene_coords = scene_coords[0].view(2, -1)
            gt_coords = gt_coords[0].view(2, -1).cuda()
            gt_coords_mask = gt_coords.abs().sum(0) > 0
            loss = torch.norm(scene_coords - gt_coords, dim=0, p=2)[gt_coords_mask]
            loss = loss.mean()
            num_valid_sc = gt_coords_mask.float().mean()
            epoch_loss += loss.item()
        else:
            # Placeholder for other modes
            loss = compute_loss_somehow()
            epoch_loss += loss.item()

        # Backpropagation and optimization
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        print('Iteration: %6d, Loss: %.7f, Time: %.2fs' %
              (iteration, loss.item(), time.time()-start_time), flush=True)
        iteration += 1
        del loss

    # Record mean loss per epoch
    average_epoch_loss = epoch_loss / num_batches
    loss_values.append(average_epoch_loss)
    print('Epoch %d, Average Loss: %.7f' % (epoch, average_epoch_loss))
    train_log.write('Epoch %d, Average Loss: %f\n' % (epoch, average_epoch_loss))

    # Save model state
    torch.save(network.state_dict(), opt.network)

# -------------------------------------------------------
# Plot and save loss curve
# -------------------------------------------------------
plt.figure()
plt.plot(loss_values, label='Average Loss per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss vs Epochs')
plt.legend()
plt.savefig(f'loss_curve_{opt.scene}_{opt.session}.png')
plt.show()

print('Saving snapshot of the network to %s.' % opt.network)
print('Done without errors.')
train_log.close()
