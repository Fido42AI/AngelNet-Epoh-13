# intention_core.py
# -*- coding: utf-8 -*-
# IntentionCore Module for AngelNet
# Author: Fedorenko Bohdan (Angel42)
# Updated by: Grok CPT-Chat
# Version: 13.7.15 (Added resonance_factor to update)

import torch

class IntentionCore:
    """
    IntentionCore module for managing the direction of intention in AngelNet.
    """
    def __init__(self, output_dim):
        """
        Initialize the IntentionCore module.
        
        Args:
            output_dim (int): Dimension of the direction vector.
        """
        self.output_dim = output_dim
        self.direction = torch.zeros(output_dim)
        self.success_history = []

    def update(self, output, success, resonance_factor=0.0):
        """
        Update the direction based on the output and success, with resonance factor.
        
        Args:
            output (torch.Tensor): Output tensor to guide the direction.
            success (bool): Whether the action was successful.
            resonance_factor (float): Resonance factor for boosting updates.
        """
        self.success_history.append(success)
        alpha_base = 0.1
        # Adjust alpha based on resonance
        resonance_boost = 1.0 + resonance_factor  # Increase effect if resonance is high
        alpha = alpha_base * resonance_boost
        self.direction = (1.0 - alpha) * self.direction + alpha * output.mean(dim=0)
        print(f"[IntentionCore] Updated direction, norm: {self.direction.norm().item():.4f}, alpha: {alpha:.4f}")

    def get_direction(self):
        """
        Get the current direction of intention.
        
        Returns:
            torch.Tensor: Current direction tensor.
        """
        return self.direction