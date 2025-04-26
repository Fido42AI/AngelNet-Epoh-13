# universal_transformer.py
# -*- coding: utf-8 -*-
# Universal Transformer Module for AngelNet
# Author: Fedorenko Bohdan (Angel42)
# Version: 13.7.8 (Added segmentation and decoding)

import torch
import torch.nn as nn
import os

class UniversalTransformer(nn.Module):
    """
    Universal Transformer module for transforming and decoding different types of data.
    """
    def __init__(self, output_dim, epsilon=1e-6, archive_dir="transformer_archive"):
        """
        Initialize the UniversalTransformer module.
        
        Args:
            output_dim (int): Output dimensionality of the transformed vectors.
            epsilon (float): Small constant to avoid division by zero.
            archive_dir (str): Directory to save/load transformer segments.
        """
        super(UniversalTransformer, self).__init__()
        self.output_dim = output_dim
        self.epsilon = epsilon
        self.archive_dir = archive_dir
        self.transformers = {}  # Dictionary of transformation matrices
        self.decoders = {}  # Dictionary of decoding matrices
        
        # Create archive directory if it doesn't exist
        if not os.path.exists(self.archive_dir):
            os.makedirs(self.archive_dir)
            print(f"[UniversalTransformer] Created archive directory: {self.archive_dir}")

    def add_data_type(self, data_type, input_dim):
        """
        Add a new data type with its corresponding transformation and decoding matrices.
        
        Args:
            data_type (str): Type of data (e.g., 'text', 'image', 'audio').
            input_dim (int): Input dimensionality of the data type.
        """
        if data_type not in self.transformers:
            # Transformation matrix (encodes data into vector)
            self.transformers[data_type] = nn.Linear(input_dim, self.output_dim)
            # Decoding matrix (decodes vector back to data approximation)
            self.decoders[data_type] = nn.Linear(self.output_dim, input_dim)
            print(f"[UniversalTransformer] Added transformer and decoder for data type '{data_type}' (input_dim={input_dim})")
            
            # Try to load from archive
            self.load_segment(data_type)

    def transform(self, data, data_type):
        """
        Transform input data into a vector.
        
        Args:
            data (torch.Tensor): Input data tensor.
            data_type (str): Type of data.
        
        Returns:
            torch.Tensor: Transformed vector.
        """
        if data_type not in self.transformers:
            raise ValueError(f"Data type '{data_type}' not supported. Add it using add_data_type().")
        
        transformer = self.transformers[data_type]
        vector = transformer(data)
        
        data_norm = data.norm().item()
        energy_scale = data_norm / (data_norm + self.epsilon)
        vector = vector * energy_scale
        
        print(f"[UniversalTransformer] Transformed {data_type} data, energy scale: {energy_scale:.4f}")
        return vector

    def decode(self, vector, data_type):
        """
        Decode a vector back to an approximation of the original data.
        
        Args:
            vector (torch.Tensor): Vector to decode.
            data_type (str): Type of data.
        
        Returns:
            torch.Tensor: Decoded data approximation.
        """
        if data_type not in self.decoders:
            raise ValueError(f"Data type '{data_type}' not supported for decoding.")
        
        decoder = self.decoders[data_type]
        decoded_data = decoder(vector)
        print(f"[UniversalTransformer] Decoded vector for data type '{data_type}'")
        return decoded_data

    def save_segment(self, data_type):
        """
        Save the transformer and decoder for a specific data type to the archive.
        
        Args:
            data_type (str): Type of data.
        """
        if data_type in self.transformers:
            checkpoint = {
                'transformer_state': self.transformers[data_type].state_dict(),
                'decoder_state': self.decoders[data_type].state_dict()
            }
            file_path = os.path.join(self.archive_dir, f"{data_type}_segment.pth")
            torch.save(checkpoint, file_path)
            print(f"[UniversalTransformer] Saved segment for data type '{data_type}' to {file_path}")

    def load_segment(self, data_type):
        """
        Load the transformer and decoder for a specific data type from the archive.
        
        Args:
            data_type (str): Type of data.
        """
        file_path = os.path.join(self.archive_dir, f"{data_type}_segment.pth")
        if os.path.exists(file_path):
            checkpoint = torch.load(file_path)
            if data_type in self.transformers:
                self.transformers[data_type].load_state_dict(checkpoint['transformer_state'])
                self.decoders[data_type].load_state_dict(checkpoint['decoder_state'])
                print(f"[UniversalTransformer] Loaded segment for data type '{data_type}' from {file_path}")
            else:
                print(f"[UniversalTransformer] Segment found for '{data_type}', but data type not initialized")

    def parameters(self):
        params = []
        for transformer in self.transformers.values():
            params.extend(transformer.parameters())
        for decoder in self.decoders.values():
            params.extend(decoder.parameters())
        return params