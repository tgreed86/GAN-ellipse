#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 10:36:08 2024

@author: trevorreed
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np



def generate_true_ellipse_points(x_center, y_center, a, b, n_points):
    theta = np.linspace(0, 2 * np.pi, n_points)
    x = x_center + a * np.cos(theta)
    y = y_center + b * np.sin(theta)
    return x, y

# Example parameters
x_true, y_true = generate_true_ellipse_points(0, 0, 3, 2, 100)


class Generator(nn.Module):
    def __init__(self, input_size, output_size):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, input_size):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


# GAN parameters
input_size = 2
generator = Generator(input_size, 2)
discriminator = Discriminator(input_size)

# Loss and optimizer
criterion = nn.BCELoss()
optimizer_G = optim.Adam(generator.parameters(), lr=0.001)
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.001)


# Training parameters
batch_size = 100
num_epochs = 50000

# Training loop
for epoch in range(num_epochs):
    # Generate random points on the ellipse
    noise = torch.randn(batch_size, input_size)
    fake_points = generator(noise)

    # Prepare real points
    real_points = torch.Tensor(list(zip(x_true, y_true)))

    # Labels for real and fake data
    real_labels = torch.ones(batch_size, 1)
    fake_labels = torch.zeros(batch_size, 1)

    # Train discriminator
    optimizer_D.zero_grad()
    real_outputs = discriminator(real_points[:batch_size])
    fake_outputs = discriminator(fake_points.detach())
    d_loss_real = criterion(real_outputs, real_labels)
    d_loss_fake = criterion(fake_outputs, fake_labels)
    d_loss = d_loss_real + d_loss_fake
    d_loss.backward()
    optimizer_D.step()

    # Train generator
    optimizer_G.zero_grad()
    fake_outputs = discriminator(fake_points)
    g_loss = criterion(fake_outputs, real_labels)
    g_loss.backward()
    optimizer_G.step()

    # Print progress
    if epoch % 1000 == 0:
        print(f"Epoch [{epoch}/{num_epochs}], D Loss: {d_loss.item()}, G Loss: {g_loss.item()}")

# Plot the generated points
generated_points = fake_points.detach().numpy()
plt.scatter(generated_points[:, 0], generated_points[:, 1], label='Generated Points')
plt.scatter(x_true, y_true, label='True Ellipse Points')
plt.legend()
plt.show()





