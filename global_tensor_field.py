# global_tensor_field.py
# -*- coding: utf-8 -*-
# GlobalTensorField Module for AngelNet
# Author: Fedorenko Bohdan (Angel42)
# Updated by: Grok CPT-Chat
# Version: 13.7.15 (Added curvature and resonance factor to update)

import torch

class GlobalTensorField:
    """
    GlobalTensorField module for maintaining a global tensor field.
    """
    def __init__(self, output_dim):
        """
        Initialize the GlobalTensorField module.
        
        Args:
            output_dim (int): Dimension of the tensor field.
        """
        self.output_dim = output_dim
        self.field = torch.zeros(output_dim)
        self.energy = 0.0

    def update(self, field, curvature, resonance_factor):
        """
        Update the global field with a new field, considering curvature and resonance.
        
        Args:
            field (torch.Tensor): New field tensor to incorporate.
            curvature (float): Curvature of the data space.
            resonance_factor (float): Resonance factor for boosting updates.
        """
        # Base learning rate
        alpha_base = 0.1
        
        # Adjust alpha based on curvature and resonance
        curvature_factor = 1.0 / (1.0 + curvature)  # Reduce effect if curvature is high
        resonance_boost = 1.0 + resonance_factor    # Increase effect if resonance is high
        alpha = alpha_base * curvature_factor * resonance_boost
        
        # Update the field
        self.field = (1.0 - alpha) * self.field + alpha * field
        
        # Update energy (using curvature as a proxy for energy)
        self.energy = curvature_factor
        print(f"[GlobalTensorField] Updated field, norm: {self.field.norm().item():.4f}, energy: {self.energy:.4f}, alpha: {alpha:.4f}")

    def get_field(self):
        """
        Get the current global field.
        
        Returns:
            torch.Tensor: Current global field tensor.
        """
        return self.field

    def get_energy(self):
        """
        Get the current energy of the field.
        
        Returns:
            float: Current energy value.
        """
        return self.energy