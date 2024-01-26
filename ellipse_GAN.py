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
import os

print(torch.version.cuda)
#device = torch.device('cuda')
#print(torch.__config__.show())
#os.environ['CUDA_HOME'] = '/u/home/reedtg/.conda/envs/ds_env/'
#cuda_home = os.environ.get('CUDA_HOME')
#print(f"CUDA_HOME is set to: {cuda_home}")
'''
# Set the correct path to the CUDA Toolkit libraries
cuda_lib_path = '/u/home/reedtg/.conda/envs/py39_GPU/'

# Set LD_LIBRARY_PATH environment variable
os.environ['LD_LIBRARY_PATH'] = f"{cuda_lib_path}:{os.environ.get('LD_LIBRARY_PATH', '')}"

# Additional check for debugging
print(f"LD_LIBRARY_PATH is set to: {os.environ.get('LD_LIBRARY_PATH')}")
'''
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
d_losses_real = []
d_losses_fake = []
g_losses = []
grad_norms_discriminator = []
grad_norms_generator = []
real_accuracies = []
fake_accuracies = []
saved_epochs = []

# GAN parameters
input_size = 2
generator = Generator(input_size, 2)
discriminator = Discriminator(input_size)
'''
if "CUDA_VISIBLE_DEVICES" in os.environ and os.environ["CUDA_VISIBLE_DEVICES"]:
    gpu_available = True
else:
    gpu_available = False
print(f"GPU available: {gpu_available}")
'''
# Move models and data to GPU
if torch.cuda.is_available():
    device = torch.device('cuda')
    print("Moving models to GPU")
else:
    device = torch.device('cpu')
    print("GPU not available. Using CPU.")

generator.to(device)
discriminator.to(device)

# Loss and optimizer
criterion = nn.BCELoss()
optimizer_G = optim.Adam(generator.parameters(), lr=0.001)
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.001)

# Training parameters
batch_size = 512
num_epochs = args.num_epochs

# Create 4 x 4 subplot to show ellipse results at various epochs of training
n_columns = 4
n_rows = 4
n_ellipse_by_epoch_plots = n_columns * n_rows
ellipse_by_epoch_fig, ellipse_by_epoch_ax = plt.subplots(nrows=n_rows, ncols=n_columns, figsize=(12, 12))
ellipse_by_epoch_ax = ellipse_by_epoch_ax.ravel()
ellipse_by_epoch_fig.suptitle('Trained Generator Ellipse by Epoch', fontsize=20)
epochs_per_plot = int(num_epochs / n_ellipse_by_epoch_plots)
#print(epochs_per_plot)
ellipse_subplot_num = 0

# Start measuring time
start_time = time.time()

# Training loop
for epoch in range(num_epochs):
    # Generate random points on the ellipse
    #noise = torch.randn(batch_size, input_size)
    noise = torch.randn(batch_size, input_size).to(device)
    fake_points = generator(noise)

    # Prepare real points
    #real_points = torch.Tensor(list(zip(x_true, y_true)))

    # Prepare real points (using a random subset of true ellipse points)
    indices = np.random.choice(len(x_true), size=batch_size, replace=False)
    real_points = torch.Tensor(list(zip(x_true[indices], y_true[indices])))

    # Move data to GPU
    fake_points = fake_points.to(device)
    real_points = real_points.to(device)

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

    # Calculate accuracy for real points
    real_predictions = (real_outputs > 0.5).float()   # Value is 1 for discriminator prediction of "real"
    real_accuracy = torch.mean((real_predictions == real_labels).float())

    # Calculate accuracy for fake points
    fake_predictions = (fake_outputs <= 0.5).float()   # Value is 0 for discriminator prediction of "fake"
    fake_accuracy = torch.mean((fake_predictions == fake_labels).float())

    # Train generator
    optimizer_G.zero_grad()
    fake_outputs = discriminator(fake_points)
    g_loss = criterion(fake_outputs, real_labels)
    g_loss.backward()
    optimizer_G.step()

    if epoch % 10 == 0:
        # Append losses to the lists
        d_losses.append(d_loss.item())
        d_losses_real.append(d_loss_real.item())
        d_losses_fake.append(d_loss_fake.item())
        g_losses.append(g_loss.item())

        # Calculate and append gradient norms
        grad_norm_discriminator = get_gradient_norm(discriminator)
        grad_norm_generator = get_gradient_norm(generator)
        grad_norms_discriminator.append(grad_norm_discriminator)
        grad_norms_generator.append(grad_norm_generator)

        # Append Accuracies to the lists
        real_accuracies.append(real_accuracy)
        fake_accuracies.append(fake_accuracy)

        # Save the current epoch number
        saved_epochs.append(epoch)

    # Print progress
    if epoch % 1000 == 0:
        print(f"Epoch [{epoch}/{num_epochs}], D Loss: {d_loss.item()}, G Loss: {g_loss.item()}")
        
    # Generate and plot results from partially trained generator 
    if (epoch % epochs_per_plot == 0 and ellipse_subplot_num < n_ellipse_by_epoch_plots):
        num_generated_points_temp = 100
        new_noise_temp = torch.randn(num_generated_points_temp, input_size)
        generated_points_temp = generator(new_noise_temp).detach().numpy()

        #Plot results
        ellipse_by_epoch_ax[ellipse_subplot_num].scatter(generated_points_temp[:, 0], generated_points_temp[:, 1], label='Generated Points')
        ellipse_by_epoch_ax[ellipse_subplot_num].scatter(x_true, y_true, label='True Ellipse Points')
        ellipse_by_epoch_ax[ellipse_subplot_num].legend()
        ellipse_by_epoch_ax[ellipse_subplot_num].set_title("Epochs: {}".format(epoch))

        ellipse_subplot_num += 1
    

# Stop measuring time 
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Training Execution Time: {elapsed_time} seconds")

# Save the ellipse subplots which show how results change with epoch
ellipse_by_epoch_fig.tight_layout()
ellipse_by_epoch_fig.savefig(outdir + "GAN_ellipse_by_epoch.pdf", format="pdf", bbox_inches="tight")

# Generate new points after training
num_generated_points = 500
new_noise = torch.randn(num_generated_points, input_size)
generated_points = generator(new_noise).detach().numpy()

# Plot the generated points
final_GAN_ellipse_fig, final_GAN_ellipse_ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
final_GAN_ellipse_ax.scatter(generated_points[:, 0], generated_points[:, 1], label='Generated Points')
final_GAN_ellipse_ax.scatter(x_true, y_true, label='True Ellipse Points')
final_GAN_ellipse_ax.legend()
final_GAN_ellipse_fig.savefig(outdir + "GAN_ellipse.pdf", format="pdf", bbox_inches="tight")


metrics_fig, metrics_ax = plt.subplots(nrows=1, ncols=3, figsize=(24, 8))
metrics_ax = metrics_ax.ravel()
metrics_fig.suptitle('Metrics Plots', fontsize=20)

# Plot loss curves
metrics_ax[0].plot(saved_epochs, d_losses, label='Discriminator Loss (total)', alpha=0.7)
metrics_ax[0].plot(saved_epochs, d_losses_real, label='Discriminator Loss (real)', alpha=0.7)
metrics_ax[0].plot(saved_epochs, d_losses_fake, label='Discriminator Loss (fake)', alpha=0.7)
metrics_ax[0].plot(saved_epochs, g_losses, label='Generator Loss', alpha=0.7)
metrics_ax[0].legend()
metrics_ax[0].set_xlabel('Epoch')
metrics_ax[0].set_ylabel('Loss')
metrics_ax[0].set_title('Loss Curves')


# Plot gradient norms
metrics_ax[1].plot(saved_epochs, grad_norms_discriminator, label='Discriminator Gradient Norm', alpha=0.7)
metrics_ax[1].plot(saved_epochs, grad_norms_generator, label='Generator Gradient Norm', alpha=0.7)
metrics_ax[1].legend()
metrics_ax[1].set_xlabel('Epoch')
metrics_ax[1].set_ylabel('Gradient Norm')
metrics_ax[1].set_title('Gradient Norms')

# Plot Accuracy of Discriminator
metrics_ax[2].plot(saved_epochs, real_accuracies, label='Discriminator Accuracy (real)', alpha=0.7)
metrics_ax[2].plot(saved_epochs, fake_accuracies, label='Discriminator Accuracy (fake)', alpha=0.7)
metrics_ax[2].legend()
metrics_ax[2].set_xlabel('Epoch')
metrics_ax[2].set_ylabel('Accuracy')
metrics_ax[2].set_title('Discriminator Accuracy')

metrics_fig.savefig(outdir + "metrics.pdf", format="pdf", bbox_inches="tight")



