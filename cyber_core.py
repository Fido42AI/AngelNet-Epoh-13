# cyber_core.py
# -*- coding: utf-8 -*-
# CyberCore Module for AngelNet
# Author: Fedorenko Bohdan (Angel42)
# Updated by: Grok CPT-Chat
# Version: 13.7.1 (Added input_dim and output_dim support)

import torch

class CyberCore:
    """
    CyberCore module for processing input and output data in the system.
    """
    def __init__(self, input_dim, output_dim):
        """
        Initialize the CyberCore module.
        
        Args:
            input_dim (int): Input dimensionality of the network.
            output_dim (int): Output dimensionality of the network.
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_state = torch.zeros(input_dim)  # Состояние входных данных
        self.output_state = torch.zeros(output_dim)  # Состояние выходных данных
        self.state = 0.0  # Скалярное значение состояния
    
    def update(self, input_data, output_data):
        """
        Update the CyberCore state based on input and output data.
        
        Args:
            input_data (torch.Tensor): Input tensor.
            output_data (torch.Tensor): Output tensor.
        """
        # Обновляем состояние входных данных
        input_data = input_data.view(-1, self.input_dim)
        self.input_state = 0.9 * self.input_state + 0.1 * input_data.mean(dim=0)
        
        # Обновляем состояние выходных данных
        self.output_state = 0.9 * self.output_state + 0.1 * output_data.mean(dim=0)
        
        # Вычисляем общее состояние как комбинацию норм
        input_norm = self.input_state.norm().item()
        output_norm = self.output_state.norm().item()
        self.state = input_norm + output_norm
        
        print(f"[CyberCore] Processed data, norm: {input_norm + output_norm:.4f}, new state: {self.state:.4f}")
    
    def summary(self):
        """
        Print a summary of the CyberCore state.
        """
        print(f"[CyberCore] State: {self.state:.4f}")