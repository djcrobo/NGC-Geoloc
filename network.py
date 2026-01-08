import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModel

class ResidualBlock(nn.Module):
    """Standard residual block with two convolution layers and a skip connection."""
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = self.relu(out)
        return out


class Network(nn.Module):
    """Scene coordinate regression network with DINOv2 backbone and residual decoder."""
    OUTPUT_SUBSAMPLE = 8  # Output resolution ratio (1/8 of input size)

    def __init__(self, mean, tiny=False):
        super(Network, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load pretrained DINOv2 model
        model_folder = './dinov2_large'
        self.dino_processor = AutoImageProcessor.from_pretrained(
            model_folder,
            do_resize=False,
            do_rescale=False
        )
        self.dino_model = AutoModel.from_pretrained(model_folder).to(self.device).eval()

        # Freeze DINOv2 parameters
        for param in self.dino_model.parameters():
            param.requires_grad = False

        # Extract DINOv2 feature dimension
        dinov2_output_dim = self.dino_model.config.hidden_size

        # Channel reduction (512 for normal, 128 for tiny)
        self.channel_reduction = nn.Conv2d(dinov2_output_dim, (512, 128)[tiny], kernel_size=1).to(self.device)

        # Residual feature refinement
        self.res_block1 = ResidualBlock((512, 128)[tiny])
        self.res_block2 = ResidualBlock((512, 128)[tiny])

        # Decoder: upsample + convolutional layers
        self.decoder = nn.Sequential(
            nn.Conv2d((512, 128)[tiny], (512, 128)[tiny], kernel_size=3, padding=1),
            nn.BatchNorm2d((512, 128)[tiny]),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),

            nn.Conv2d((512, 128)[tiny], (256, 64)[tiny], kernel_size=3, padding=1),
            nn.BatchNorm2d((256, 64)[tiny]),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),

            nn.Conv2d((256, 64)[tiny], (128, 32)[tiny], kernel_size=3, padding=1),
            nn.BatchNorm2d((128, 32)[tiny]),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),

            nn.Conv2d((128, 32)[tiny], 2, kernel_size=1)  # 2 channels for (x, y)
        )

        # Register mean offset
        if isinstance(mean, torch.Tensor):
            self.register_buffer('mean', mean.clone().detach())
        else:
            self.register_buffer('mean', torch.tensor(mean, dtype=torch.float32))

        self.tiny = tiny

    def forward(self, inputs):
        """
        Args:
            inputs: [B,1,448,672] grayscale images
        Returns:
            [B,2,56,84] predicted (x, y) coordinates
        """
        inputs = inputs.to(self.device)
        patch_size = self.dino_model.config.patch_size  # typically 14
        H, W = inputs.shape[2], inputs.shape[3]

        # Convert grayscale to RGB
        rgb_inputs = inputs.repeat(1, 3, 1, 1)

        # DINOv2 preprocessing
        processed_inputs = self.dino_processor(images=rgb_inputs, return_tensors="pt").pixel_values.to(self.device)

        # DINOv2 feature extraction
        with torch.no_grad():
            dino_outputs = self.dino_model(processed_inputs)
            last_hidden_state = dino_outputs.last_hidden_state  # [B, T, C]

        # Reshape tokens into feature map
        num_patches_h, num_patches_w = H // patch_size, W // patch_size
        expected_tokens = num_patches_h * num_patches_w
        B, T, C = last_hidden_state.size()

        if T == expected_tokens:
            feature_map = last_hidden_state.transpose(1, 2).reshape(B, C, num_patches_h, num_patches_w)
        elif T == expected_tokens + 1:
            token_features = last_hidden_state[:, 1:, :]
            feature_map = token_features.transpose(1, 2).reshape(B, C, num_patches_h, num_patches_w)
        elif T == 257:
            token_features = last_hidden_state[:, 1:, :]
            feature_map = token_features.transpose(1, 2).reshape(B, C, 16, 16)
        else:
            raise ValueError(f"Unexpected token count: {T}")

        # Channel reduction
        feature_map = self.channel_reduction(feature_map)

        # Residual refinement
        res = self.res_block1(feature_map)
        res = F.relu(res + feature_map)
        res = self.res_block2(res)
        res = F.relu(res + feature_map)

        # Decode to coordinate field
        output_coords = self.decoder(res)

        # Downsample to final resolution
        output_coords = F.interpolate(output_coords, size=(56, 84), mode='bilinear', align_corners=False)

        # Apply mean offset
        output_coords = output_coords + self.mean.view(1, 2, 1, 1)
        return output_coords
