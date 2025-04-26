# curvature_movie.py
# -*- coding: utf-8 -*-
# CurvatureMovie Module for AngelNet
# Author: Fedorenko Bohdan (Angel42)
# Updated by: Grok CPT-Chat
# Version: 13.7.1 (Added output_dim support)

import torch
import torch.nn.functional as F

class CurvatureMovie:
    """
    CurvatureMovie module for tracking curvature dynamics in the system.
    """
    def __init__(self, output_dim):
        """
        Initialize the CurvatureMovie module.
        
        Args:
            output_dim (int): Output dimensionality of the network.
        """
        self.output_dim = output_dim
        self.frames = []  # Список кадров (тензоры)
        self.max_frames = 100  # Максимальное количество кадров
        self.avg_curvature = 0.0
        self.avg_diff_norm = 0.0
        self.avg_data_norm = 0.0
        self.avg_field_norm = 0.0
    
    def compute_curvature(self, output):
        """
        Compute the curvature (1 - CosSim) between the current output and the last frame.
        
        Args:
            output (torch.Tensor): Output tensor from the network.
        
        Returns:
            float: Curvature value (1 - CosSim).
        """
        if not self.frames:
            return 0.0
        
        last_frame = self.frames[-1]
        # Убедимся, что размерности совпадают
        if last_frame.size(0) != output.size(0):
            print(f"[CurvatureMovie] Warning: Output size {output.size()} does not match last frame size {last_frame.size()}")
            return 0.0
        
        # Вычисляем косинусную схожесть
        cos_sim = F.cosine_similarity(output, last_frame, dim=1).mean().item()
        curvature = 1.0 - cos_sim
        return curvature
    
    def update(self, output, curvature):
        """
        Update the movie by adding a new frame and computing metrics.
        
        Args:
            output (torch.Tensor): Output tensor from the network.
            curvature (float): Precomputed curvature value.
        """
        # Добавляем новый кадр
        self.frames.append(output.clone().detach())
        if len(self.frames) > self.max_frames:
            self.frames.pop(0)
        
        # Вычисляем метрики
        data_norm = output.norm().item()
        
        # Для field_norm используем среднее последнего кадра (заглушка)
        field = output.mean(dim=0)
        field_norm = field.norm().item()
        
        # Вычисляем разницу между кадрами (Diff Norm)
        diff_norm = 0.0
        if len(self.frames) > 1:
            diff = self.frames[-1] - self.frames[-2]
            diff_norm = diff.norm().item()
        
        # Обновляем средние значения
        n = len(self.frames)
        self.avg_curvature = (self.avg_curvature * (n - 1) + curvature) / n if n > 0 else 0.0
        self.avg_diff_norm = (self.avg_diff_norm * (n - 1) + diff_norm) / n if n > 0 else 0.0
        self.avg_data_norm = (self.avg_data_norm * (n - 1) + data_norm) / n if n > 0 else 0.0
        self.avg_field_norm = (self.avg_field_norm * (n - 1) + field_norm) / n if n > 0 else 0.0
        
        print(f"[CurvatureMovie] Added frame - Data Norm: {data_norm:.4f}, Field Norm: {field_norm:.4f}, "
              f"Curvature (1 - CosSim): {curvature:.4f}, Diff Norm: {diff_norm:.4f}")
    
    def summary(self):
        """
        Print a summary of the curvature movie state.
        """
        print(f"[CurvatureMovie] Frames: {len(self.frames)}")
        print(f"Average Curvature (1 - CosSim): {self.avg_curvature:.4f}, "
              f"Average Diff Norm: {self.avg_diff_norm:.4f}, "
              f"Average Data Norm: {self.avg_data_norm:.4f}, "
              f"Average Field Norm: {self.avg_field_norm:.4f}")