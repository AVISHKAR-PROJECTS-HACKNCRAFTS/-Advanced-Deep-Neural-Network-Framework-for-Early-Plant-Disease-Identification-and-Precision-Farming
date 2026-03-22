"""
CNN module for plant disease classification.

Defines a custom 4-block convolutional neural network that classifies
leaf images into one of 39 disease/healthy classes across 14 plant species.
The model expects 224x224 RGB input images and outputs logits for 39 classes.

Architecture summary:
    - 4 convolutional blocks, each: Conv2d -> ReLU -> BatchNorm2d -> Conv2d -> ReLU -> BatchNorm2d -> MaxPool2d
    - Channel progression: 3 -> 32 -> 64 -> 128 -> 256
    - Dense head: Flatten(50176) -> Dropout(0.4) -> Linear(1024) -> ReLU -> Dropout(0.4) -> Linear(K)
    - K = 39 output classes (default)

Trained on the Plant Village Dataset (~61,486 images).
"""

import pandas as pd
import torch.nn as nn


class CNN(nn.Module):
    """
    Custom 4-block CNN for plant disease classification.

    Each convolutional block consists of two Conv2d layers with ReLU activations
    and BatchNorm2d normalization, followed by MaxPool2d for spatial downsampling.
    The classifier head uses two fully connected layers with dropout regularization.

    Args:
        K (int): Number of output classes. Default for this project is 39.

    Input:
        Tensor of shape (batch_size, 3, 224, 224) -- RGB images resized to 224x224.

    Output:
        Tensor of shape (batch_size, K) -- raw logits (unnormalized scores) for each class.
    """

    idx_to_classes = {
        0: 'Apple___Apple_scab',
        1: 'Apple___Black_rot',
        2: 'Apple___Cedar_apple_rust',
        3: 'Apple___healthy',
        4: 'Background_without_leaves',
        5: 'Blueberry___healthy',
        6: 'Cherry___Powdery_mildew',
        7: 'Cherry___healthy',
        8: 'Corn___Cercospora_leaf_spot Gray_leaf_spot',
        9: 'Corn___Common_rust',
        10: 'Corn___Northern_Leaf_Blight',
        11: 'Corn___healthy',
        12: 'Grape___Black_rot',
        13: 'Grape___Esca_(Black_Measles)',
        14: 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
        15: 'Grape___healthy',
        16: 'Orange___Haunglongbing_(Citrus_greening)',
        17: 'Peach___Bacterial_spot',
        18: 'Peach___healthy',
        19: 'Pepper,_bell___Bacterial_spot',
        20: 'Pepper,_bell___healthy',
        21: 'Potato___Early_blight',
        22: 'Potato___Late_blight',
        23: 'Potato___healthy',
        24: 'Raspberry___healthy',
        25: 'Soybean___healthy',
        26: 'Squash___Powdery_mildew',
        27: 'Strawberry___Leaf_scorch',
        28: 'Strawberry___healthy',
        29: 'Tomato___Bacterial_spot',
        30: 'Tomato___Early_blight',
        31: 'Tomato___Late_blight',
        32: 'Tomato___Leaf_Mold',
        33: 'Tomato___Septoria_leaf_spot',
        34: 'Tomato___Spider_mites Two-spotted_spider_mite',
        35: 'Tomato___Target_Spot',
        36: 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
        37: 'Tomato___Tomato_mosaic_virus',
        38: 'Tomato___healthy',
    }

    def __init__(self, K):
        super(CNN, self).__init__()
        self.conv_layers = nn.Sequential(
            # conv1
            nn.Conv2d(in_channels=3, out_channels=32,
                      kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32,
                      kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            # conv2
            nn.Conv2d(in_channels=32, out_channels=64,
                      kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=64,
                      kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            # conv3
            nn.Conv2d(in_channels=64, out_channels=128,
                      kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(in_channels=128, out_channels=128,
                      kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
            # conv4
            nn.Conv2d(in_channels=128, out_channels=256,
                      kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(in_channels=256, out_channels=256,
                      kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2),
        )

        self.dense_layers = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(50176, 1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, K),
        )

    def forward(self, X):
        out = self.conv_layers(X)

        # Flatten
        out = out.view(-1, 50176)

        # Fully connected
        out = self.dense_layers(out)

        return out

    @staticmethod
    def validate_input(x):
        """Validate input tensor has correct shape (batch, 3, 224, 224)."""
        if x.dim() != 4:
            raise ValueError(f"Expected 4D tensor, got {x.dim()}D")
        if x.shape[1] != 3:
            raise ValueError(f"Expected 3 channels, got {x.shape[1]}")
        if x.shape[2] != 224 or x.shape[3] != 224:
            raise ValueError(f"Expected 224x224, got {x.shape[2]}x{x.shape[3]}")
        return True

    @staticmethod
    def get_class_name(index):
        """Get human-readable class name for a prediction index."""
        if index in CNN.idx_to_classes:
            return CNN.idx_to_classes[index]
        return f"Unknown class (index {index})"


# Module-level alias for backward compatibility.
idx_to_classes = CNN.idx_to_classes
