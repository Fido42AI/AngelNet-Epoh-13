# action_signal.py
# -*- coding: utf-8 -*-
# ActionSignalLayer v13.7 — Action Signal Dynamics
# Author: Fedorenko Bohdan (Angel42)
# Updated by: Grok CPT-Chat
# Version: 13.7.2 (Fixed Output Dimension)

import torch
import torch.nn as nn
import logging

logging.basicConfig(
    filename='action_signal.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class ActionSignalLayer(nn.Module):
    def __init__(self, output_dim=10):  # Изменили default на 10
        super(ActionSignalLayer, self).__init__()
        self.output_dim = output_dim
        self.signal = torch.zeros(output_dim, dtype=torch.float32)
        self.signal_history = []
        self.max_history = 10

    def update(self, data, intention_direction, resonance_factor=0.0):
        # Update signal based on data and intention
        signal_update = data.mean(dim=0) if data.dim() > 1 else data
        signal_update = signal_update[:self.output_dim] if signal_update.size(0) >= self.output_dim else signal_update.repeat(self.output_dim)[:self.output_dim]
        self.signal = 0.9 * self.signal + 0.1 * (signal_update + intention_direction)
        
        # Усиливаем сигнал на основе резонанса
        self.signal *= (1.0 + resonance_factor)
        logging.info(f"Signal boosted by resonance, factor: {resonance_factor:.4f}, new norm: {self.signal.norm().item():.4f}")
        
        self.signal_history.append(self.signal.clone())
        if len(self.signal_history) > self.max_history:
            self.signal_history.pop(0)
        
        # Создаём резонансный вектор для обратной связи
        feedback_resonance = self.signal * 0.5
        return feedback_resonance

    def summary(self):
        signal_norm = self.signal.norm().item()
        print(f"[ActionSignalLayer] Signal Norm: {signal_norm:.4f}")