# Ultralytics YOLO 🚀, AGPL-3.0 license
"""Lightweight neural network adaptor module for feature refinement."""

import torch.nn as nn


class LightweightAdaptor(nn.Module):
    """A lightweight neural network adaptor module for feature refinement.

    This module implements a lightweight adaptation mechanism that allows for feature refinement
    while maintaining computational efficiency. It uses a bottleneck architecture with
    channel reduction and projection.

    Args:
        in_channels (int): Number of input channels.
        reduction_ratio (int, optional): Channel reduction ratio for the bottleneck.
            Default is 32.
        out_channels (int, optional): Number of output channels. If None, same as in_channels.

    Attributes:
        down_proj (nn.Conv2d): 1x1 convolution for channel reduction
        act (nn.ReLU): ReLU activation function
        up_proj (nn.Conv2d): 1x1 convolution for channel projection

    Notes:
        - The module uses residual connection (identity + transformation)
        - Up projection weights are initialized to zero to start with identity mapping
        - Channel reduction helps in reducing computational cost
    """

    def __init__(self, in_channels, reduction_ratio=32, out_channels=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels
        hidden_channels = max(1, in_channels // reduction_ratio)

        self.proj = None
        if self.in_channels != self.out_channels:
            self.proj = nn.Conv2d(in_channels, self.out_channels, 1)
            nn.init.zeros_(self.proj.weight)
            nn.init.zeros_(self.proj.bias)

        self.down_proj = nn.Conv2d(self.in_channels, hidden_channels, 1)
        self.act = nn.ReLU(inplace=True)
        self.up_proj = nn.Conv2d(hidden_channels, self.out_channels, 1)

        # Initialize to zero
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)

    def forward(self, x):
        identity = x if self.proj is None else self.proj(x)
        out = self.down_proj(x)
        out = self.act(out)
        out = self.up_proj(out)
        return identity + out
