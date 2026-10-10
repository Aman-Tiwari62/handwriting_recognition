import torch
import torch.nn as nn

from models.cnn import CNNEncoder


model = CNNEncoder()

x = torch.randn(4, 1, 64, 279)

# CNN
features = model(x)

print("CNN output shape:", features.shape)


# [B, C, H, W]
B, C, H, W = features.shape

# Move width to the sequence dimension
features = features.permute(0, 3, 1, 2)

print("After permute:", features.shape)


# Combine C and H
features = features.reshape(B, W, C * H)

print("After reshape:", features.shape)


# Project 4096 features → 256 features
projection = nn.Linear(C * H, 256)

features = projection(features)

print("After projection:", features.shape)