# tensor_global_vector_map.py
# -*- coding: utf-8 -*-
# TensorGlobalVectorMap Module for AngelNet
# Author: Fedorenko Bohdan (Angel42)
# Updated by: Grok CPT-Chat
# Version: 13.7.14 (Added persistent vector history, duplicate checking, data compression, fixed entropy calculation)

import torch
import torch.nn.functional as F
import os
import gzip
import pickle

class TensorGlobalVectorMap:
    """
    TensorGlobalVectorMap module for managing class-specific tensor fields and vector interpretation.
    """
    def __init__(self, output_dim, num_classes=10, gravity_scale=0.1, epsilon=1e-6, memory_file="class_fields.pth", vector_history_file="vector_history.pth"):
        self.output_dim = output_dim
        self.num_classes = num_classes
        self.vectors = []
        self.max_vectors = 100
        self.vector_history = []  # Full history of vectors to be saved to file
        self.class_fields = [torch.zeros(output_dim) for _ in range(num_classes)]
        self.class_counts = [0 for _ in range(num_classes)]
        self.resonance_vector = torch.zeros(output_dim)
        self.resonance_factor = 0.0
        self.gravity_scale = gravity_scale
        self.epsilon = epsilon
        self.memory_file = memory_file
        self.vector_history_file = vector_history_file
        
        self.base_vectors = {}
        self.base_vector_counts = {}
        self.base_vector_alpha = 0.1
        self.familiarity_threshold = 0.8
        
        self.load_memory()
        self.load_vector_history()
    def load_memory(self):
        if os.path.exists(self.memory_file):
            with gzip.open(self.memory_file, 'rb') as f:
                checkpoint = pickle.load(f)
            self.class_fields = checkpoint['class_fields']
            self.class_counts = checkpoint['class_counts']
            self.base_vectors = checkpoint.get('base_vectors', {})
            self.base_vector_counts = checkpoint.get('base_vector_counts', {})
            print(f"[TensorGlobalVectorMap] Loaded memory from {self.memory_file}")
        else:
            print(f"[TensorGlobalVectorMap] No memory file found, starting fresh")

    def load_vector_history(self):
        """
        Load the vector history from the vector history file.
        """
        if os.path.exists(self.vector_history_file):
            with gzip.open(self.vector_history_file, 'rb') as f:
                checkpoint = pickle.load(f)
            encoded_history = checkpoint.get('vector_history', [])
            
            # Decode the history
            self.vector_history = []
            prev_vector = None
            for encoded_vector in encoded_history:
                decoded_vector = self.decode_delta_vector(encoded_vector, prev_vector)
                self.vector_history.append(decoded_vector)
                prev_vector = decoded_vector
            
            print(f"[TensorGlobalVectorMap] Loaded {len(self.vector_history)} vectors from {self.vector_history_file}")
            
            # Populate in-memory vectors with the last max_vectors items
            self.vectors = self.vector_history[-self.max_vectors:]
        else:
            print(f"[TensorGlobalVectorMap] No vector history file found at {self.vector_history_file}, starting fresh")

    def save_memory(self):
        checkpoint = {
            'class_fields': self.class_fields,
            'class_counts': self.class_counts,
            'base_vectors': self.base_vectors,
            'base_vector_counts': self.base_vector_counts
        }
        with gzip.open(self.memory_file, 'wb') as f:
            pickle.dump(checkpoint, f)
        print(f"[TensorGlobalVectorMap] Saved memory to {self.memory_file}")

    def save_vector_history(self):
        """
        Save the full vector history to the vector history file.
        """
        # Encode the history using Delta Encoding
        encoded_history = []
        prev_vector = None
        for vector in self.vector_history:
            encoded_vector = self.encode_delta_vector(vector, prev_vector)
            encoded_history.append(encoded_vector)
            prev_vector = vector
        
        checkpoint = {
            'vector_history': encoded_history
        }
        with gzip.open(self.vector_history_file, 'wb') as f:
            pickle.dump(checkpoint, f)
        print(f"[TensorGlobalVectorMap] Saved {len(self.vector_history)} vectors to {self.vector_history_file}")

    def encode_delta_vector(self, vector, prev_vector=None):
        """
        Encode a vector using Delta Encoding.
        
        Args:
            vector (torch.Tensor): Vector to encode.
            prev_vector (torch.Tensor, optional): Previous vector for delta encoding.
        
        Returns:
            torch.Tensor: Encoded vector.
        """
        if prev_vector is None:
            return vector.clone()
        else:
            delta = vector - prev_vector
            delta = torch.clamp(delta * 255, -128, 127).to(torch.int8)
            return delta

    def decode_delta_vector(self, encoded_vector, prev_vector=None):
        """
        Decode a vector using Delta Decoding.
        
        Args:
            encoded_vector (torch.Tensor): Encoded vector to decode.
            prev_vector (torch.Tensor, optional): Previous vector for delta decoding.
        
        Returns:
            torch.Tensor: Decoded vector.
        """
        if prev_vector is None:
            return encoded_vector.clone()
        else:
            delta = encoded_vector.to(torch.float32) / 255.0
            return prev_vector + delta

    def is_vector_duplicate(self, new_vector):
        """
        Check if the new vector is a duplicate of the last vector in vector_history.
        
        Args:
            new_vector (torch.Tensor): New vector to check.
        
        Returns:
            bool: True if the vector is a duplicate, False otherwise.
        """
        if not self.vector_history:
            return False
        
        last_vector = self.vector_history[-1]
        cos_sim = F.cosine_similarity(new_vector.unsqueeze(0), last_vector.unsqueeze(0), dim=1).item()
        return cos_sim > 0.9999

    def update_base_vector(self, vector, data_type):
        vector = vector.view(-1)[:self.output_dim]
        
        if data_type not in self.base_vectors:
            self.base_vectors[data_type] = vector.clone()
            self.base_vector_counts[data_type] = 1
            print(f"[TensorGlobalVectorMap] Initialized base vector for data type '{data_type}'")
        else:
            self.base_vectors[data_type] = (1.0 - self.base_vector_alpha) * self.base_vectors[data_type] + self.base_vector_alpha * vector
            self.base_vector_counts[data_type] += 1
            print(f"[TensorGlobalVectorMap] Updated base vector for data type '{data_type}', count: {self.base_vector_counts[data_type]}")
            
    def get_base_vector(self, data_type):
        return self.base_vectors.get(data_type, torch.zeros(self.output_dim))

    def is_familiar(self, vector, data_type):
        vector = vector.view(-1)[:self.output_dim]
        base_vector = self.get_base_vector(data_type)
        
        if base_vector.norm().item() == 0:
            return False
        
        cos_sim = F.cosine_similarity(vector.unsqueeze(0), base_vector.unsqueeze(0), dim=1).item()
        is_familiar = cos_sim > self.familiarity_threshold
        print(f"[TensorGlobalVectorMap] Vector familiarity for {data_type}: {cos_sim:.4f}, familiar: {is_familiar}")
        return is_familiar

    def interpret_vector(self, vector, data_type):
        """
        Interpret the vector by decomposing it into contributions from base vectors and class fields.
        
        Args:
            vector (torch.Tensor): Input vector.
            data_type (str): Type of data.
        
        Returns:
            dict: Interpretation of the vector.
        """
        vector = vector.view(-1)[:self.output_dim]
        interpretation = {
            'base_vector_contribution': 0.0,
            'class_contributions': [],
            'entropy': 0.0
        }
        
        # Contribution from base vector
        base_vector = self.get_base_vector(data_type)
        if base_vector.norm().item() > 0:
            base_cos_sim = F.cosine_similarity(vector.unsqueeze(0), base_vector.unsqueeze(0), dim=1).item()
            interpretation['base_vector_contribution'] = base_cos_sim
        else:
            interpretation['base_vector_contribution'] = 0.0
        
        # Contributions from class fields
        similarities = []
        for class_id in range(self.num_classes):
            field = self.class_fields[class_id]
            if self.class_counts[class_id] == 0:
                similarities.append(0.0)
                continue
            cos_sim = F.cosine_similarity(vector.unsqueeze(0), field.unsqueeze(0), dim=1).item()
            similarities.append(max(0.0, cos_sim))
        
        # Normalize similarities to get a probability distribution
        similarities_sum = sum(similarities) + self.epsilon
        probs = [sim / similarities_sum for sim in similarities]
        interpretation['class_contributions'] = probs
        
        # Compute entropy to measure uncertainty
        # Collect terms as tensors and sum them using torch.sum
        terms = [p * torch.log(torch.tensor(p + self.epsilon)) for p in probs if p > 0]
        entropy = -torch.sum(torch.stack(terms)) if terms else torch.tensor(0.0)
        interpretation['entropy'] = entropy.item()
        
        print(f"[TensorGlobalVectorMap] Interpreted vector: base_contribution={interpretation['base_vector_contribution']:.4f}, "
              f"class_contributions={interpretation['class_contributions']}, entropy={interpretation['entropy']:.4f}")
        return interpretation
    def apply_tensor_gravity(self):
        if len(self.vectors) < 2:
            return
        
        new_vectors = [v.clone() for v in self.vectors]
        
        for i in range(len(self.vectors)):
            total_force = torch.zeros_like(self.vectors[i])
            v_i = self.vectors[i]
            mass_i = v_i.norm().item()
            
            for j in range(len(self.vectors)):
                if i == j:
                    continue
                v_j = self.vectors[j]
                mass_j = v_j.norm().item()
                
                cos_sim = F.cosine_similarity(v_i.unsqueeze(0), v_j.unsqueeze(0), dim=1).item()
                distance = 1.0 - cos_sim + self.epsilon
                
                force_magnitude = self.gravity_scale * (mass_i * mass_j) / (distance ** 2)
                direction = (v_j - v_i)
                norm = direction.norm().item()
                if norm > 0:
                    direction = direction / norm
                else:
                    direction = torch.zeros_like(direction)
                
                total_force += force_magnitude * direction
            
            new_vectors[i] = v_i + total_force
        
        self.vectors = new_vectors

    def update(self, vector, label, data_type, confidence=1.0, sigma=1.0, curvature=0.0, resonance_factor=0.0):
        vector = vector.view(-1)[:self.output_dim]
        
        # Check for duplicates
        if self.is_vector_duplicate(vector):
            print(f"[TensorGlobalVectorMap] Skipped duplicate vector for label {label}")
            return
        
        self.vectors.append(vector)
        if len(self.vectors) > self.max_vectors:
            self.vectors.pop(0)
        
        # Add to full vector history
        self.vector_history.append(vector)
        self.save_vector_history()
        
        self.apply_tensor_gravity()
        
        self.update_base_vector(vector, data_type)
        
        if 0 <= label < self.num_classes:
            old_field = self.class_fields[label]
            diff = vector - old_field
            kernel = torch.exp(-diff.norm().pow(2) / (2 * sigma ** 2)).item()
            energy = vector.norm().item()
            
            alpha_base = 0.1 * confidence * kernel * energy
            curvature_factor = 1.0 / (1.0 + curvature)
            resonance_boost = 1.0 + resonance_factor
            alpha = alpha_base * curvature_factor * resonance_boost
            
            self.class_fields[label] = (1.0 - alpha) * self.class_fields[label] + alpha * vector
            self.class_counts[label] += 1
            print(f"[TensorGlobalVectorMap] Updated field for class {label}, count: {self.class_counts[label]}, kernel: {kernel:.4f}, energy: {energy:.4f}, alpha: {alpha:.4f}")

    def classify(self, vector):
        vector = vector.view(-1)[:self.output_dim]
        similarities = []
        
        for class_id in range(self.num_classes):
            field = self.class_fields[class_id]
            if self.class_counts[class_id] == 0:
                similarities.append(-float('inf'))
                continue
            cos_sim = F.cosine_similarity(vector.unsqueeze(0), field.unsqueeze(0), dim=1).item()
            similarities.append(cos_sim)
        
        predicted_class = similarities.index(max(similarities))
        print(f"[TensorGlobalVectorMap] Classified vector as class {predicted_class}, similarities: {similarities}")
        return predicted_class

    def cluster_vectors(self, threshold=0.1):
        for class_id in range(self.num_classes):
            if self.class_counts[class_id] < 2:
                continue
            
            class_vectors = [v for v in self.vectors if self.classify(v) == class_id]
            if len(class_vectors) < 2:
                continue
            
            clusters = []
            for v in class_vectors:
                assigned = False
                for cluster in clusters:
                    centroid = torch.mean(torch.stack(cluster), dim=0)
                    cos_sim = F.cosine_similarity(v.unsqueeze(0), centroid.unsqueeze(0), dim=1).item()
                    if cos_sim > threshold:
                        cluster.append(v)
                        assigned = True
                        break
                if not assigned:
                    clusters.append([v])
            
            if clusters:
                largest_cluster = max(clusters, key=len)
                self.class_fields[class_id] = torch.mean(torch.stack(largest_cluster), dim=0)
                print(f"[TensorGlobalVectorMap] Clustered vectors for class {class_id}, new field norm: {self.class_fields[class_id].norm().item():.4f}")

    def get_field(self, label=None):
        if label is not None and 0 <= label < self.num_classes:
            return self.class_fields[label]
        total_count = sum(self.class_counts)
        if total_count == 0:
            return torch.zeros(self.output_dim)
        weighted_field = sum(f * c for f, c in zip(self.class_fields, self.class_counts))
        return weighted_field / total_count

    def get_resonance_vector(self):
        return self.resonance_vector

    def get_resonance_factor(self):
        if len(self.vectors) < 2:
            print("[TensorGlobalVectorMap] Not enough vectors for resonance factor, returning 0.0")
            return 0.0
        
        v1, v2 = self.vectors[-2], self.vectors[-1]
        if v1.size() != torch.Size([self.output_dim]) or v2.size() != torch.Size([self.output_dim]):
            print(f"[TensorGlobalVectorMap] Vector size mismatch: v1 {v1.size()}, v2 {v2.size()}, expected {self.output_dim}")
            return 0.0
        
        cos_sim = F.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0), dim=1).item()
        self.resonance_factor = 1.0 - cos_sim
        self.resonance_vector = v2.clone()
        return self.resonance_factor

    def summary(self):
        print(f"[TensorGlobalVectorMap] Vectors: {len(self.vectors)}, Full Vector History: {len(self.vector_history)}, Resonance Factor: {self.resonance_factor:.4f}")    


            