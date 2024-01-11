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
import time
import argparse

#Get arguments 
parser = argparse.ArgumentParser()
parser.add_argument("num_epochs", help="Enter number of times dataset should be trained over (epochs)", type=int)
args = parser.parse_args()

outdir = "/work/clas12/reedtg/data_science/GAN_ellipse_example/plots/"

def generate_true_ellipse_points(x_center, y_center, a, b, n_points):
    theta = np.linspace(0, 2 * np.pi, n_points)
    x = x_center + a * np.cos(theta)
    y = y_center + b * np.sin(theta)
    return x, y

# Get gradients during training
def get_gradient_norm(model):
    total_norm = 0
    for param in model.parameters():
        if param.grad is not None:
            total_norm += param.grad.data.norm(2).item()
    return total_norm

# Example parameters
x_true, y_true = generate_true_ellipse_points(0, 0, 3, 2, 128)


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


# Generate true ellipse points
x_true, y_true = generate_true_ellipse_points(0, 0, 3, 2, 512)

# Lists to store loss values and gradients
d_losses = []
g_losses = []
grad_norms_discriminator = []
grad_norms_generator = []

# GAN parameters
input_size = 2
generator = Generator(input_size, 2)
discriminator = Discriminator(input_size)

# Loss and optimizer
criterion = nn.BCELoss()
optimizer_G = optim.Adam(generator.parameters(), lr=0.001)
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.001)


# Training parameters
batch_size = 64
num_epochs = args.num_epochs

# Start measuring time
start_time = time.time()

# Training loop
for epoch in range(num_epochs):
    # Generate random points on the ellipse
    noise = torch.randn(batch_size, input_size)
    fake_points = generator(noise)

    # Prepare real points
    #real_points = torch.Tensor(list(zip(x_true, y_true)))

    # Prepare real points (using a random subset of true ellipse points)
    indices = np.random.choice(len(x_true), size=batch_size, replace=False)
    real_points = torch.Tensor(list(zip(x_true[indices], y_true[indices])))

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

    if epoch % 10 == 0:
        # Append losses to the lists
        d_losses.append(d_loss.item())
        g_losses.append(g_loss.item())

        # Calculate and append gradient norms
        grad_norm_discriminator = get_gradient_norm(discriminator)
        grad_norm_generator = get_gradient_norm(generator)
        grad_norms_discriminator.append(grad_norm_discriminator)
        grad_norms_generator.append(grad_norm_generator)

    # Print progress
    if epoch % 1000 == 0:
        print(f"Epoch [{epoch}/{num_epochs}], D Loss: {d_loss.item()}, G Loss: {g_loss.item()}")


# Stop measuring time 
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Training Execution Time: {elapsed_time} seconds")

#generated_points = fake_points.detach().numpy()

# Generate new points after training
num_generated_points = 500
new_noise = torch.randn(num_generated_points, input_size)
generated_points = generator(new_noise).detach().numpy()

# Plot the generated points
plt.scatter(generated_points[:, 0], generated_points[:, 1], label='Generated Points')
plt.scatter(x_true, y_true, label='True Ellipse Points')
plt.legend()
#plt.show()
plt.savefig(outdir + "GAN_ellipse.pdf", format="pdf", bbox_inches="tight")


metrics_fig, metrics_ax = plt.subplots(nrows=1, ncols=2, figsize=(18, 8))
metrics_ax = metrics_ax.ravel()
metrics_fig.suptitle('Metrics Plots', fontsize=20)

# Plot loss curves
metrics_ax[0].plot(d_losses, label='Discriminator Loss', alpha=0.7)
metrics_ax[0].plot(g_losses, label='Generator Loss', alpha=0.7)
metrics_ax[0].legend()
metrics_ax[0].set_xlabel('Epoch')
metrics_ax[0].set_ylabel('Loss')
metrics_ax[0].set_title('Loss Curves')


# Plot gradient norms
metrics_ax[1].plot(grad_norms_discriminator, label='Discriminator Gradient Norm', alpha=0.7)
metrics_ax[1].plot(grad_norms_generator, label='Generator Gradient Norm', alpha=0.7)
metrics_ax[1].legend()
metrics_ax[1].set_xlabel('Epoch')
metrics_ax[1].set_ylabel('Gradient Norm')
metrics_ax[1].set_title('Gradient Norms')

metrics_fig.savefig(outdir + "metrics.pdf", format="pdf", bbox_inches="tight")



