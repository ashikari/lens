from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class CNNConfig:
    """Configuration for the SimpleCNN model."""

    in_channels: int = 1
    cnn_channels: List[int] = None  # [4, 8] for two conv layers
    kernel_size: int = 3
    stride: int = 2
    padding: int = 1
    mlp_layers: List[int] = None  # [12, 10] for two linear layers
    dropout_p: float = 0.25
    input_size: int = 28

    def __post_init__(self):
        if self.cnn_channels is None:
            self.cnn_channels = [4, 8]
        if self.mlp_layers is None:
            self.mlp_layers = [12, 10]


# Configuration presets for different model sizes
def create_nano_config():
    """~6k parameters"""
    return CNNConfig(cnn_channels=[4, 8], mlp_layers=[12, 10])


def create_half_config():
    """~30k parameters"""
    return CNNConfig(cnn_channels=[8, 16], mlp_layers=[30, 10])


def create_matching_config():
    """~60k parameters"""
    return CNNConfig(cnn_channels=[12, 24], mlp_layers=[50, 10])


def create_double_config():
    """~120k parameters"""
    return CNNConfig(cnn_channels=[18, 36], mlp_layers=[100, 10])


def create_quad_config():
    """~240k parameters"""
    return CNNConfig(cnn_channels=[24, 48], mlp_layers=[180, 10])


def create_tenx_config():
    """~600k parameters"""
    return CNNConfig(cnn_channels=[32, 64], mlp_layers=[350, 10])


# Default configuration (nano)
CNN_CONFIG = create_nano_config()

CNN_CONFIGS = {
    "nano": create_nano_config(),
    "half": create_half_config(),
    "matching": create_matching_config(),
    "double": create_double_config(),
    "quad": create_quad_config(),
    "tenx": create_tenx_config(),
}


class SimpleCNN(nn.Module):
    """
    A simple CNN for MNIST digit classification.

    Architecture:
    - Input: (batch_size, 1, 28, 28) - MNIST images
    - Multiple 3x3 convolutional layers with ReLU activation and stride=2
    - Flatten the output
    - Multiple linear layers with ReLU activation in between
    - Output: (batch_size, num_classes) - classification output
    """

    def __init__(self, config: CNNConfig = None):
        super(SimpleCNN, self).__init__()

        # Use provided config or default to CNN_CONFIG
        self.config = config if config is not None else CNN_CONFIG

        # Calculate the size after strided convolutions
        def conv_out_size(size, kernel_size, stride, padding):
            return (size + 2 * padding - kernel_size) // stride + 1

        # Calculate spatial dimensions after each conv layer
        current_size = self.config.input_size
        for _ in self.config.cnn_channels:
            current_size = conv_out_size(current_size, self.config.kernel_size, self.config.stride, self.config.padding)

        self.flattened_size = self.config.cnn_channels[-1] * current_size * current_size

        # Create convolutional layers
        self.conv_layers = nn.ModuleList()
        in_channels = self.config.in_channels

        for out_channels in self.config.cnn_channels:
            conv_layer = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=self.config.kernel_size,
                stride=self.config.stride,
                padding=self.config.padding,
            )
            self.conv_layers.append(conv_layer)
            in_channels = out_channels

        # Create MLP layers
        self.mlp_layers = nn.ModuleList()
        mlp_input_size = self.flattened_size

        for i, out_features in enumerate(self.config.mlp_layers):
            linear_layer = nn.Linear(mlp_input_size, out_features)
            self.mlp_layers.append(linear_layer)
            mlp_input_size = out_features

        # Dropout for regularization
        self.dropout = nn.Dropout(self.config.dropout_p)

    def forward(self, x):
        # Input shape: (batch_size, 1, 28, 28)

        # Apply convolutional layers
        for conv_layer in self.conv_layers:
            x = F.relu(conv_layer(x))

        batch_size = x.size(0)
        x = x.view(batch_size, self.flattened_size)

        # Apply MLP layers
        for i, linear_layer in enumerate(self.mlp_layers):
            x = linear_layer(x)
            # Apply ReLU and dropout to all layers except the last one
            if i < len(self.mlp_layers) - 1:
                x = F.relu(x)
                x = self.dropout(x)

        return x


def test_all_configs():
    """Test all configurations and display their parameter counts."""

    print("=" * 60)
    print("CONFIGURATION PARAMETER COUNTS")
    print("=" * 60)

    for name, config in CNN_CONFIGS.items():
        model = SimpleCNN(config)
        total_params = sum(p.numel() for p in model.parameters())

        print(f"\n{name.upper():<10}: {total_params:>8,} parameters")
        print(f"  CNN channels: {config.cnn_channels}")
        print(f"  MLP layers: {config.mlp_layers}")

        # Test forward pass
        batch_size = 2
        sample_input = torch.randn(batch_size, 1, 28, 28)
        output = model(sample_input)
        expected_size = config.mlp_layers[-1]

        if output.shape == (batch_size, expected_size):
            print(f"  ✅ Forward pass successful: {output.shape}")
        else:
            print(f"  ❌ Forward pass failed: expected ({batch_size}, {expected_size}), got {output.shape}")


if __name__ == "__main__":
    test_all_configs()
