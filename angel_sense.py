# angel_sense.py
# -*- coding: utf-8 -*-
# AngelSense Module for AngelNet
# Author: Fedorenko Bohdan (Angel42)
# Updated by: Grok CPT-Chat
# Version: 13.7.13 (Added duplicate checking and data compression)

import torch
import torch.nn.functional as F
import os
import gzip
import pickle
from collections import deque

class AngelSense:
    """
    AngelSense module for sensing and analyzing tensor fields, curvature, and resonance.
    """
    def __init__(self, max_history=100, history_file="sense_history.pth"):
        """
        Initialize the AngelSense module.
        
        Args:
            max_history (int): Maximum number of sense states to keep in memory.
            history_file (str): File to store the persistent history.
        """
        self.max_history = max_history
        self.history_file = history_file
        self.sense_history = deque(maxlen=max_history)  # In-memory history (limited)
        self.full_history = []  # Full history to be saved to file
        self.epsilon = 1e-5  # Tolerance for float comparison
        
        # Load history from file if it exists
        self.load_history()

    def quantize_float(self, value):
        """
        Quantize a float value (0-1) to int8 (0-255).
        
        Args:
            value (float): Value to quantize.
        
        Returns:
            int: Quantized value.
        """
        return int(min(max(value, 0.0), 1.0) * 255)

    def dequantize_float(self, value):
        """
        Dequantize an int8 value (0-255) to float (0-1).
        
        Args:
            value (int): Quantized value.
        
        Returns:
            float: Dequantized value.
        """
        return value / 255.0

    def encode_delta(self, new_state, prev_state=None):
        """
        Encode a state using Delta Encoding.
        
        Args:
            new_state (dict): New state to encode.
            prev_state (dict, optional): Previous state for delta encoding.
        
        Returns:
            dict: Encoded state.
        """
        encoded = {}
        
        if prev_state is None:
            # First state: store as is (quantized)
            encoded['field'] = new_state['field'].clone()
            encoded['curvature'] = self.quantize_float(new_state['curvature'])
            encoded['resonance_factor'] = self.quantize_float(new_state['resonance_factor'])
        else:
            # Compute delta for field
            delta_field = new_state['field'] - prev_state['field']
            # Quantize delta to int8 (-128 to 127)
            delta_field = torch.clamp(delta_field * 255, -128, 127).to(torch.int8)
            encoded['field'] = delta_field
            
            # Delta for curvature and resonance_factor
            delta_curvature = self.quantize_float(new_state['curvature']) - self.quantize_float(prev_state['curvature'])
            delta_resonance = self.quantize_float(new_state['resonance_factor']) - self.quantize_float(prev_state['resonance_factor'])
            encoded['curvature'] = delta_curvature
            encoded['resonance_factor'] = delta_resonance
        
        return encoded
    def decode_delta(self, encoded_state, prev_state=None):
        """
        Decode a state using Delta Decoding.
        
        Args:
            encoded_state (dict): Encoded state to decode.
            prev_state (dict, optional): Previous state for delta decoding.
        
        Returns:
            dict: Decoded state.
        """
        decoded = {}
        
        if prev_state is None:
            # First state: decode directly
            decoded['field'] = encoded_state['field'].clone()
            decoded['curvature'] = self.dequantize_float(encoded_state['curvature'])
            decoded['resonance_factor'] = self.dequantize_float(encoded_state['resonance_factor'])
        else:
            # Decode field
            delta_field = encoded_state['field'].to(torch.float32) / 255.0
            decoded['field'] = prev_state['field'] + delta_field
            
            # Decode curvature and resonance_factor
            decoded_curvature = self.dequantize_float(prev_state['curvature']) + (encoded_state['curvature'] / 255.0)
            decoded_resonance = self.dequantize_float(prev_state['resonance_factor']) + (encoded_state['resonance_factor'] / 255.0)
            decoded['curvature'] = min(max(decoded_curvature, 0.0), 1.0)
            decoded['resonance_factor'] = min(max(decoded_resonance, 0.0), 1.0)
        
        return decoded

    def load_history(self):
        """
        Load the sense history from the history file with gzip compression.
        """
        if os.path.exists(self.history_file):
            with gzip.open(self.history_file, 'rb') as f:
                checkpoint = pickle.load(f)
            encoded_history = checkpoint.get('sense_history', [])
            
            # Decode the history
            self.full_history = []
            prev_state = None
            for encoded_state in encoded_history:
                decoded_state = self.decode_delta(encoded_state, prev_state)
                self.full_history.append(decoded_state)
                prev_state = decoded_state
            
            print(f"[AngelSense] Loaded {len(self.full_history)} sense states from {self.history_file}")
            
            # Populate in-memory history with the last max_history items
            for state in self.full_history[-self.max_history:]:
                self.sense_history.append(state)
        else:
            print(f"[AngelSense] No history file found at {self.history_file}, starting fresh")

    def save_history(self):
        """
        Save the full sense history to the history file with gzip compression.
        """
        # Encode the history using Delta Encoding
        encoded_history = []
        prev_state = None
        for state in self.full_history:
            encoded_state = self.encode_delta(state, prev_state)
            encoded_history.append(encoded_state)
            prev_state = state
        
        checkpoint = {
            'sense_history': encoded_history
        }
        with gzip.open(self.history_file, 'wb') as f:
            pickle.dump(checkpoint, f)
        print(f"[AngelSense] Saved {len(self.full_history)} sense states to {self.history_file}")

    def is_duplicate(self, new_state):
        """
        Check if the new state is a duplicate of the last state in full_history.
        
        Args:
            new_state (dict): New sense state to check.
        
        Returns:
            bool: True if the state is a duplicate, False otherwise.
        """
        if not self.full_history:
            return False
        
        last_state = self.full_history[-1]
        
        # Compare fields using cosine similarity
        cos_sim = F.cosine_similarity(new_state['field'].unsqueeze(0), last_state['field'].unsqueeze(0), dim=1).item()
        if cos_sim < 0.9999:  # Not identical
            return False
        
        # Compare curvature and resonance factor
        curvature_diff = abs(new_state['curvature'] - last_state['curvature'])
        resonance_diff = abs(new_state['resonance_factor'] - last_state['resonance_factor'])
        
        if curvature_diff > self.epsilon or resonance_diff > self.epsilon:
            return False
        
        return True

    def sense(self, field, curvature, resonance_factor):
        """
        Sense the current field state, store it in history if not a duplicate, and save to file.
        
        Args:
            field (torch.Tensor): Tensor field to sense.
            curvature (float): Curvature of the field.
            resonance_factor (float): Resonance factor.
        """
        # Create the sense state
        sense_state = {
            'field': field.clone(),
            'curvature': curvature,
            'resonance_factor': resonance_factor
        }
        
        # Check for duplicates
        if self.is_duplicate(sense_state):
            print(f"[AngelSense] Skipped duplicate sense state: curvature={curvature:.4f}, resonance_factor={resonance_factor:.4f}")
            return
        
        # Add to in-memory history (limited by max_history)
        self.sense_history.append(sense_state)
        
        # Add to full history (unlimited, for saving to file)
        self.full_history.append(sense_state)
        
        # Save the full history to file
        self.save_history()
        
        print(f"[AngelSense] Sensed field: curvature={curvature:.4f}, resonance_factor={resonance_factor:.4f}, history_size={len(self.sense_history)}")

    def get_average_curvature(self):
        """
        Compute the average curvature from the in-memory sense history.
        
        Returns:
            float: Average curvature, or 0.0 if history is empty.
        """
        if not self.sense_history:
            return 0.0
        avg_curvature = sum(state['curvature'] for state in self.sense_history) / len(self.sense_history)
        return avg_curvature

    def get_average_resonance(self):
        """
        Compute the average resonance factor from the in-memory sense history.
        
        Returns:
            float: Average resonance factor, or 0.0 if history is empty.
        """
        if not self.sense_history:
            return 0.0
        avg_resonance = sum(state['resonance_factor'] for state in self.sense_history) / len(self.sense_history)
        return avg_resonance

    def summary(self):
        """
        Print a summary of the current in-memory sense history.
        """
        avg_curvature = self.get_average_curvature()
        avg_resonance = self.get_average_resonance()
        print(f"[AngelSense] Summary: in_memory_history_size={len(self.sense_history)}, full_history_size={len(self.full_history)}, avg_curvature={avg_curvature:.4f}, avg_resonance={avg_resonance:.4f}")    