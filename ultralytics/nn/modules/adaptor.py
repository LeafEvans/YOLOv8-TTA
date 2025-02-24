# Ultralytics YOLO 🚀, AGPL-3.0 license
"""Adaptor module."""

import torch.nn as nn

__all__ = "LightweightAdaptor"


class LightweightAdaptor(nn.Module):
    """A lightweight neural network adaptor module for feature refinement."""

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
