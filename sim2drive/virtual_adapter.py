import torch
import torch.nn as nn
import torch.nn.functional as F


class VirtualAdapter(nn.Module):
    """
    A bottleneck convolutional adapter that transforms input images for domain adaptation.
    Uses a bottleneck architecture: 3 channels → hidden_dim (bottleneck) → 3 channels
    Ensures output is compatible with DINOv2 (dimensions must be multiples of 14).
    """

    def __init__(self, input_dim=3, hidden_dim=64, dropout_rate=0.1, output_size=378):
        super(VirtualAdapter, self).__init__()

        # Encoder: compress to bottleneck
        self.encoder_conv1 = nn.Conv2d(input_dim, hidden_dim // 2, kernel_size=3, padding=1)
        self.encoder_bn1 = nn.BatchNorm2d(hidden_dim // 2)

        self.encoder_conv2 = nn.Conv2d(hidden_dim // 2, hidden_dim, kernel_size=3, padding=1)
        self.encoder_bn2 = nn.BatchNorm2d(hidden_dim)

        # Decoder: expand from bottleneck back to RGB
        self.decoder_conv1 = nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1)
        self.decoder_bn1 = nn.BatchNorm2d(hidden_dim // 2)

        self.decoder_conv2 = nn.Conv2d(hidden_dim // 2, input_dim, kernel_size=3, padding=1)
        self.decoder_bn2 = nn.BatchNorm2d(input_dim)

        self.dropout = nn.Dropout2d(dropout_rate)
        self.activation = nn.ReLU()

        # Output size must be multiple of 14 for DINOv2 with patch_size=14
        # Common valid sizes: 224, 378, 392, 518, 532, etc.
        self.output_size = output_size

    def forward(self, x):
        """
        Args:
            x: Input RGB image tensor of shape (B, 3, H, W)

        Returns:
            Adapted image tensor of shape (B, 3, output_size, output_size)
        """
        identity = x

        # Encoder: compress through bottleneck
        out = self.activation(self.encoder_bn1(self.encoder_conv1(x)))
        out = self.dropout(out)

        out = self.activation(self.encoder_bn2(self.encoder_conv2(out)))
        out = self.dropout(out)

        # Decoder: expand back to RGB
        out = self.activation(self.decoder_bn1(self.decoder_conv1(out)))
        out = self.dropout(out)

        out = self.decoder_bn2(self.decoder_conv2(out))

        # Residual connection (only works if identity has same spatial dims)
        if identity.shape[2:] == out.shape[2:]:
            out = out + identity

        # Resize to DINOv2-compatible size (multiple of 14)
        if out.shape[2] != self.output_size or out.shape[3] != self.output_size:
            out = F.interpolate(out, size=(self.output_size, self.output_size),
                               mode='bilinear', align_corners=False)

        return out