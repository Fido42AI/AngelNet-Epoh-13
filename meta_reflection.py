# meta_reflection.py
# -*- coding: utf-8 -*-
# MetaReflection Module for AngelNet
# Author: Fedorenko Bohdan (Angel42)
# Updated by: Grok (xAI)
# Version: 13.7.1 (Added output_dim support)

import torch

class MetaReflection:
    """
    MetaReflection module for tracking and reflecting on the system's state.
    """
    def __init__(self, output_dim):
        """
        Initialize the MetaReflection module.
        
        Args:
            output_dim (int): Output dimensionality of the network.
        """
        self.output_dim = output_dim
        self.state = torch.zeros(output_dim)  # Вектор состояния
        self.reflection_state = 0.0  # Скалярное значение рефлексии
        self.mood_norm = 0.0  # Норма настроения (будет обновляться)
    
    def update(self, output, success):
        """
        Update the reflection state based on the output and success.
        
        Args:
            output (torch.Tensor): Output tensor from the network.
            success (bool): Whether the goal was achieved.
        """
        # Обновляем вектор состояния как среднее выходов
        self.state = 0.9 * self.state + 0.1 * output.mean(dim=0)
        
        # Обновляем рефлексию (на основе нормы состояния и успеха)
        self.reflection_state = self.state.norm().item()
        if success:
            self.reflection_state += 1.0  # Увеличиваем рефлексию при успехе
        
        # Обновляем норму настроения (заглушка, может быть связана с CogniCore)
        self.mood_norm += 0.1 if success else -0.05
        self.mood_norm = max(0.0, min(self.mood_norm, 5.0))  # Ограничиваем
        
        print(f"[MetaReflection] Updated - Output Norm: {output.norm().item():.4f}, "
              f"Mood Norm: {self.mood_norm:.4f}, Reflection State: {self.reflection_state:.4f}")
    
    def summary(self):
        """
        Print a summary of the reflection state.
        """
        print(f"[MetaReflection] State: {self.reflection_state:.4f}")