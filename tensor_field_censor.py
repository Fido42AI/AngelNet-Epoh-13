# tensor_field_censor.py
# -*- coding: utf-8 -*-
# TensorFieldCensor Module for AngelNet
# Author: Fedorenko Bohdan (Angel42)
# Updated by: Grok CPT-Chat
# Version: 13.7.15 (Added duplicate checking, data compression, fixed curvature_factor and resonance_boost in apply_field_gravity)

import torch
import torch.nn.functional as F
import os
import gzip
import pickle
from collections import deque
import shutil

class TensorFieldCensor:
    """
    TensorFieldCensor module for managing and censoring tensor fields with gravity and history.
    """
    def __init__(self, output_dim, num_classes=10, gravity_scale=0.1, max_history=100, archive_dir="field_archive", ideal_fields_file="ideal_fields.pth"):
        """
        Initialize the TensorFieldCensor module.
        
        Args:
            output_dim (int): Dimension of the tensor fields.
            num_classes (int): Number of classes.
            gravity_scale (float): Scale factor for gravitational forces between fields.
            max_history (int): Maximum number of field states to keep in memory.
            archive_dir (str): Directory to store archived field states.
            ideal_fields_file (str): File to store the ideal fields.
        """
        self.output_dim = output_dim
        self.num_classes = num_classes
        self.gravity_scale = gravity_scale
        self.max_history = max_history
        self.archive_dir = archive_dir
        self.ideal_fields_file = ideal_fields_file
        self.epsilon = 1e-6
        
        self.field_history = deque(maxlen=max_history)  # In-memory history (limited)
        self.full_field_history = []  # Full history to be saved to file
        self.ideal_fields = [torch.zeros(output_dim) for _ in range(num_classes)]
        self.ideal_counts = [0 for _ in range(num_classes)]
        
        # Create archive directory if it doesn't exist
        if not os.path.exists(self.archive_dir):
            os.makedirs(self.archive_dir)
            print(f"[TensorFieldCensor] Created archive directory: {self.archive_dir}")
        
        self.load_history()
        self.load_ideal_fields()
    def load_history(self):
        """
        Load the field history from the archive directory.
        """
        history_files = [f for f in os.listdir(self.archive_dir) if f.startswith('field_state_') and f.endswith('.pth')]
        if not history_files:
            print("[TensorFieldCensor] No field history found, starting fresh")
            return
        
        latest_file = max(history_files, key=lambda f: int(f.split('_')[-1].split('.')[0]))
        latest_path = os.path.join(self.archive_dir, latest_file)
        with gzip.open(latest_path, 'rb') as f:
            checkpoint = pickle.load(f)
        self.full_field_history = checkpoint.get('field_history', [])
        print(f"[TensorFieldCensor] Loaded {len(self.full_field_history)} field states from {latest_path}")
        
        # Populate in-memory history with the last max_history items
        for state in self.full_field_history[-self.max_history:]:
            self.field_history.append(state)

    def save_history(self):
        """
        Save the full field history to the archive directory.
        """
        if not self.full_field_history:
            return
        
        timestamp = len(self.full_field_history)
        archive_path = os.path.join(self.archive_dir, f'field_state_{timestamp}.pth')
        checkpoint = {
            'field_history': self.full_field_history
        }
        with gzip.open(archive_path, 'wb') as f:
            pickle.dump(checkpoint, f)
        print(f"[TensorFieldCensor] Saved {len(self.full_field_history)} field states to {archive_path}")
        
        # Keep only the last 10 history files
        history_files = [f for f in os.listdir(self.archive_dir) if f.startswith('field_state_') and f.endswith('.pth')]
        if len(history_files) > 10:
            history_files.sort(key=lambda f: int(f.split('_')[-1].split('.')[0]))
            for old_file in history_files[:-10]:
                os.remove(os.path.join(self.archive_dir, old_file))
                print(f"[TensorFieldCensor] Removed old history file: {old_file}")

    def load_ideal_fields(self):
        """
        Load the ideal fields from the ideal fields file.
        """
        if os.path.exists(self.ideal_fields_file):
            with gzip.open(self.ideal_fields_file, 'rb') as f:
                checkpoint = pickle.load(f)
            self.ideal_fields = checkpoint['ideal_fields']
            self.ideal_counts = checkpoint['ideal_counts']
            print(f"[TensorFieldCensor] Loaded ideal fields from {self.ideal_fields_file}")
        else:
            print(f"[TensorFieldCensor] No ideal fields file found at {self.ideal_fields_file}, starting fresh")

    def save_ideal_fields(self):
        """
        Save the ideal fields to the ideal fields file.
        """
        checkpoint = {
            'ideal_fields': self.ideal_fields,
            'ideal_counts': self.ideal_counts
        }
        with gzip.open(self.ideal_fields_file, 'wb') as f:
            pickle.dump(checkpoint, f)
        print(f"[TensorFieldCensor] Saved ideal fields to {self.ideal_fields_file}")

    def quantize_float(self, value, min_val=-1.0, max_val=1.0, bits=8):
        """
        Quantize a float value to a specified number of bits.
        """
        scale = (2 ** bits - 1) / (max_val - min_val)
        value = (value - min_val) * scale
        value = torch.round(value).to(torch.int32)
        value = torch.clamp(value, 0, 2 ** bits - 1)
        return value

    def dequantize_float(self, value, min_val=-1.0, max_val=1.0, bits=8):
        """
        Dequantize a value back to a float.
        """
        scale = (max_val - min_val) / (2 ** bits - 1)
        value = value.to(torch.float32) * scale + min_val
        return value

    def encode_delta(self, vector, prev_vector=None):
        """
        Encode a vector using Delta Encoding.
        """
        if prev_vector is None:
            return vector.clone()
        delta = vector - prev_vector
        delta = torch.clamp(delta * 255, -128, 127).to(torch.int8)
        return delta

    def decode_delta(self, encoded_vector, prev_vector=None):
        """
        Decode a vector using Delta Decoding.
        """
        if prev_vector is None:
            return encoded_vector.clone()
        delta = encoded_vector.to(torch.float32) / 255.0
        return prev_vector + delta

    def is_field_duplicate(self, new_fields, curvature, resonance_factor, accuracy, mood_norm):
        """
        Check if the new field state is a duplicate of the last state in full_field_history.
        """
        if not self.full_field_history:
            return False
        
        last_state = self.full_field_history[-1]
        last_fields = last_state['fields']
        
        # Compare fields using cosine similarity
        total_cos_sim = 0.0
        for new_field, last_field in zip(new_fields, last_fields):
            cos_sim = F.cosine_similarity(new_field.unsqueeze(0), last_field.unsqueeze(0), dim=1).item()
            total_cos_sim += cos_sim
        avg_cos_sim = total_cos_sim / len(new_fields)
        if avg_cos_sim < 0.9999:  # Not identical
            return False
        
        # Compare other metrics
        curvature_diff = abs(curvature - last_state['curvature'])
        resonance_diff = abs(resonance_factor - last_state['resonance_factor'])
        accuracy_diff = abs(accuracy - last_state['accuracy'])
        mood_diff = abs(mood_norm - last_state['mood_norm'])
        
        if curvature_diff > self.epsilon or resonance_diff > self.epsilon or accuracy_diff > self.epsilon or mood_diff > self.epsilon:
            return False
        
        return True

    def apply_field_gravity(self, class_fields, curvature, resonance_factor):
        """
        Apply gravitational forces between fields to adjust their positions.
        
        Args:
            class_fields (list): List of field tensors.
            curvature (float): Curvature of the data space.
            resonance_factor (float): Resonance factor for boosting updates.
        
        Returns:
            list: Updated fields after applying gravity.
        """
        new_fields = [field.clone() for field in class_fields]
        
        # Compute scaling factors based on curvature and resonance
        curvature_factor = 1.0 / (1.0 + curvature)  # Reduce effect if curvature is high
        resonance_boost = 1.0 + resonance_factor    # Increase effect if resonance is high
        
        for i in range(len(class_fields)):
            total_force = torch.zeros_like(class_fields[i])
            field_i = class_fields[i]
            mass_i = field_i.norm().item()
            
            for j in range(len(class_fields)):
                if i == j:
                    continue
                field_j = class_fields[j]
                mass_j = field_j.norm().item()
                
                cos_sim = F.cosine_similarity(field_i.unsqueeze(0), field_j.unsqueeze(0), dim=1).item()
                distance = 1.0 - cos_sim + self.epsilon
                
                force_magnitude = self.gravity_scale * (mass_i * mass_j) / (distance ** 2)
                direction = (field_j - field_i)
                norm = direction.norm().item()
                if norm > 0:
                    direction = direction / norm
                else:
                    direction = torch.zeros_like(direction)
                
                # Apply scaling factors to the force
                scaled_force = force_magnitude * direction * curvature_factor * resonance_boost
                total_force += scaled_force
            
            new_fields[i] = field_i + total_force
        
        print(f"[TensorFieldCensor] Applied field gravity, curvature_factor={curvature_factor:.4f}, resonance_boost={resonance_boost:.4f}")
        return new_fields  
    def update(self, class_fields, label, curvature, resonance_factor, accuracy, mood_norm):
        """
        Update the field censor with new class fields and metrics.
        """
        # Apply gravity to adjust fields
        gravity_fields = self.apply_field_gravity(class_fields, curvature, resonance_factor)
        
        # Check for duplicates
        if self.is_field_duplicate(gravity_fields, curvature, resonance_factor, accuracy, mood_norm):
            print("[TensorFieldCensor] Skipped duplicate field state")
            return
        
        # Update field history
        state = {
            'fields': gravity_fields,
            'curvature': curvature,
            'resonance_factor': resonance_factor,
            'accuracy': accuracy,
            'mood_norm': mood_norm
        }
        self.field_history.append(state)
        self.full_field_history.append(state)
        print(f"[TensorFieldCensor] Added field state to history: curvature={curvature:.4f}, resonance_factor={resonance_factor:.4f}, accuracy={accuracy:.4f}, mood_norm={mood_norm:.4f}")
        
        self.save_history()
        
        # Update ideal fields
        if 0 <= label < self.num_classes:
            alpha = 0.1 * (1.0 / (1.0 + curvature)) * (1.0 + resonance_factor)
            self.ideal_fields[label] = (1.0 - alpha) * self.ideal_fields[label] + alpha * gravity_fields[label]
            self.ideal_counts[label] += 1
            print(f"[TensorFieldCensor] Updated ideal field for class {label}, count: {self.ideal_counts[label]}, alpha: {alpha:.4f}")
            self.save_ideal_fields()

    def suggest_new_direction(self, class_fields, label, curvature, resonance_factor, accuracy, mood_norm):
        """
        Suggest a new direction for the fields based on recent history.
        """
        if not self.field_history:
            return None
        
        recent_states = list(self.field_history)[-5:]  # Last 5 states
        if len(recent_states) < 2:
            return None
        
        # Compute the average direction of change in fields
        total_direction = torch.zeros_like(class_fields[0])
        for i in range(len(recent_states) - 1):
            fields_t0 = recent_states[i]['fields']
            fields_t1 = recent_states[i + 1]['fields']
            for j in range(self.num_classes):
                direction = fields_t1[j] - fields_t0[j]
                total_direction += direction
        
        total_direction /= (len(recent_states) - 1) * self.num_classes
        
        # Scale direction based on curvature, resonance, accuracy, and mood
        curvature_factor = 1.0 / (1.0 + curvature)
        resonance_boost = 1.0 + resonance_factor
        accuracy_factor = accuracy
        mood_factor = 1.0 / (1.0 + abs(mood_norm - 0.5))
        
        scaled_direction = total_direction * curvature_factor * resonance_boost * accuracy_factor * mood_factor
        print(f"[TensorFieldCensor] Suggested new direction, norm={scaled_direction.norm().item():.4f}, "
              f"curvature_factor={curvature_factor:.4f}, resonance_boost={resonance_boost:.4f}, "
              f"accuracy_factor={accuracy_factor:.4f}, mood_factor={mood_factor:.4f}")
        return scaled_direction

    def summary(self):
        """
        Print a summary of the TensorFieldCensor state.
        """
        if not self.field_history:
            print("[TensorFieldCensor] No field history available")
            return
        
        latest_state = self.field_history[-1]
        print(f"[TensorFieldCensor] Latest State: "
              f"Fields Norms: {[field.norm().item() for field in latest_state['fields']]}, "
              f"Curvature: {latest_state['curvature']:.4f}, "
              f"Resonance Factor: {latest_state['resonance_factor']:.4f}, "
              f"Accuracy: {latest_state['accuracy']:.4f}, "
              f"Mood Norm: {latest_state['mood_norm']:.4f}")    