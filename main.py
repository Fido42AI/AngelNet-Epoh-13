# main.py
# -*- coding: utf-8 -*-
# AngelNet Demo Script
# Author: Fedorenko Bohdan (Angel42)
# Updated by: Grok CPT-Chat
# Version: 13.7.8 (Added vector interpretation and decoding)

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import time

from angelnet_core import AngelNet

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Hyperparameters
input_size = 784  # 28x28
hidden_size = 128
num_classes = 10
num_epochs = 2
batch_size = 32
learning_rate = 0.001

# MNIST dataset
train_dataset = torchvision.datasets.MNIST(root='./data', train=True,
                                           transform=transforms.ToTensor(), download=True)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size,
                                           shuffle=True)

# Initialize the network
net = AngelNet(input_dim=input_size, hidden_dim=hidden_size, output_dim=num_classes, base_lr=learning_rate).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(list(net.parameters()) + list(net.transformer.parameters()), lr=learning_rate)

# Set optimizer in the network
net.set_optimizer(optimizer)

# Training loop
total_time = 0.0
total_batches = 0

for epoch in range(num_epochs):
    epoch_start = time.time()
    
    print(f"Starting Epoch {epoch + 1}")
    
    correct_total = 0
    total_samples = 0
    
    for i, (images, labels) in enumerate(train_loader):
        batch_start = time.time()
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = net(images, labels, data_type='image')
        
        # Decode the first output vector to see how the system interprets it
        if i == 0:  # Decode only for the first batch to avoid clutter
            vector = outputs[0]
            decoded_data = net.transformer.decode(vector, data_type='image')
            print(f"[Main] Decoded vector norm: {decoded_data.norm().item():.4f}")

        if net.training_mode:
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == labels).sum().item()
            accuracy = correct / batch_size
            print(f"[Debug] Batch {total_batches + 1}: Correct predictions: {correct}/{batch_size}, Accuracy: {accuracy:.2%}")
            net.think(images, labels, accuracy=accuracy, success=(accuracy > net.goal.target_accuracy), loss=loss.item(), data_type='image')
        else:
            predicted = net.autonomous_classify(images, data_type='image')
            correct = (predicted == labels).sum().item()
            accuracy = correct / batch_size
            print(f"[Debug] Batch {total_batches + 1}: Correct predictions: {correct}/{batch_size}, Accuracy: {accuracy:.2%}")
            net.think(images, labels, accuracy=accuracy, success=(accuracy > net.goal.target_accuracy), loss=0.0, data_type='image')
        
        correct_total += correct
        total_samples += batch_size
        
        batch_time = time.time() - batch_start
        print(f"Batch {total_batches + 1} took {batch_time:.4f} seconds")
        total_batches += 1
    
    epoch_accuracy = correct_total / total_samples
    print(f"Epoch {epoch + 1} accuracy: {epoch_accuracy:.2%}")
    
    net.save_memory()
    net.visualize_metrics(epoch + 1)
    epoch_time = time.time() - epoch_start
    print(f"Epoch {epoch + 1} took {epoch_time:.4f} seconds")
    total_time += epoch_time

print(f"Total time: {total_time:.4f} seconds")