# angel_graph.py
# -*- coding: utf-8 -*-
# AngelGraph Module for AngelNet
# Author: Fedorenko Bohdan (Angel42)
# Updated by: Grok CPT-Chat
# Version: 13.7.2 (Fixed direction_mean handling)

import torch

class AngelGraph:
    """
    AngelGraph module for modeling mood and output graphs in the system.
    """
    def __init__(self, num_nodes):
        """
        Initialize the AngelGraph module.
        
        Args:
            num_nodes (int): Number of nodes in the graph.
        """
        self.num_nodes = num_nodes
        self.mood_nodes = torch.zeros(num_nodes)  # Узлы настроения
        self.mood_edges = torch.zeros((num_nodes, num_nodes))  # Рёбра настроения
        self.output_nodes = torch.zeros(num_nodes)  # Узлы выходов
        self.output_edges = torch.zeros((num_nodes, num_nodes))  # Рёбра выходов
        self.avg_mood_edge_weight = 0.0
        self.avg_output_edge_weight = 0.0
    
    def update(self, output, direction):
        """
        Update the graph based on the output and direction.
        
        Args:
            output (torch.Tensor): Output tensor from the network.
            direction (torch.Tensor): Direction tensor from IntentionCore.
        """
        # Обновляем узлы настроения (заглушка, используем случайные значения)
        self.mood_nodes = torch.randn(self.num_nodes) * 0.1
        
        # Обновляем рёбра настроения (заглушка, случайная матрица смежности)
        self.mood_edges = torch.randn(self.num_nodes, self.num_nodes) * 0.5
        self.mood_edges = (self.mood_edges + self.mood_edges.t()) / 2  # Симметрическая матрица
        self.avg_mood_edge_weight = self.mood_edges.mean().item()
        
        # Обновляем узлы выходов
        output_mean = output.mean(dim=0)  # [batch_size, output_dim] -> [output_dim]
        direction_mean = direction  # direction уже имеет размер [output_dim], не усредняем
        
        # Приводим output_mean и direction_mean к размеру [num_nodes]
        if output_mean.size(0) < self.num_nodes:
            # Если размер меньше num_nodes, дополняем нули
            padding = torch.zeros(self.num_nodes - output_mean.size(0), device=output_mean.device)
            output_mean = torch.cat([output_mean, padding])
        elif output_mean.size(0) > self.num_nodes:
            # Если размер больше, обрезаем
            output_mean = output_mean[:self.num_nodes]
        
        if direction_mean.size(0) < self.num_nodes:
            padding = torch.zeros(self.num_nodes - direction_mean.size(0), device=direction_mean.device)
            direction_mean = torch.cat([direction_mean, padding])
        elif direction_mean.size(0) > self.num_nodes:
            direction_mean = direction_mean[:self.num_nodes]
        
        # Обновляем output_nodes
        self.output_nodes = output_mean + direction_mean
        
        # Обновляем рёбра выходов (заглушка, случайная матрица смежности)
        self.output_edges = torch.randn(self.num_nodes, self.num_nodes) * 0.5
        self.output_edges = (self.output_edges + self.output_edges.t()) / 2  # Симметрическая матрица
        self.avg_output_edge_weight = self.output_edges.mean().item()
        
        print(f"[AngelGraph] Built from CogniCore - Mood Nodes: {self.num_nodes}, Mood Edges: {self.num_nodes * 2}")
        print(f"[AngelGraph] Updated with output - Output Nodes: {self.num_nodes}, Output Edges: {self.num_nodes * 2}")
        print(f"Average Mood Edge Weight: {self.avg_mood_edge_weight:.4f}")
        print(f"Average Output Edge Weight: {self.avg_output_edge_weight:.4f}")
    
    def summary(self):
        """
        Print a summary of the graph state.
        """
        print(f"[AngelGraph] Mood Nodes: {self.num_nodes}, Mood Edges: {self.num_nodes * 2}")
        print(f"[AngelGraph] Output Nodes: {self.num_nodes}, Output Edges: {self.num_nodes * 2}")