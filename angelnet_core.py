# angelnet_core.py
# -*- coding: utf-8 -*-
# AngelNet Core Module
# Author: Fedorenko Bohdan (Angel42)
# Updated by: Grok CPT-Chat
# Version: 13.7.15 (Fixed GlobalTensorField update arguments)

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from cyber_core import CyberCore
from curvature_movie import CurvatureMovie
from tensor_global_vector_map import TensorGlobalVectorMap
from global_tensor_field import GlobalTensorField
from intention_core import IntentionCore
from action_signal import ActionSignalLayer
from meta_reflection import MetaReflection
from angel_graph import AngelGraph
from angel_goal_module import AngelGoalModule
from angel_sense import AngelSense
from cogni_core import CogniCore
from universal_transformer import UniversalTransformer
from tensor_field_censor import TensorFieldCensor

class AngelNet(nn.Module):
    """
    AngelNet module combining a neural network with dynamic components.
    """
    def __init__(self, input_dim=784, hidden_dim=128, output_dim=10, base_lr=0.001):
        super(AngelNet, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.base_lr = base_lr
        self.num_classes = output_dim
        
        self.transformer = UniversalTransformer(output_dim=output_dim)
        self.transformer.add_data_type('image', input_dim)
        
        self.fc1 = nn.Linear(output_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        
        self.global_field = GlobalTensorField(output_dim=output_dim)
        self.intention = IntentionCore(output_dim=output_dim)
        self.action = ActionSignalLayer(output_dim=output_dim)
        self.vector_map = TensorGlobalVectorMap(output_dim=output_dim, num_classes=output_dim, gravity_scale=0.1, vector_history_file="vector_history.pth")
        self.reflection = MetaReflection(output_dim=output_dim)
        self.graph = AngelGraph(num_nodes=32)
        self.movie = CurvatureMovie(output_dim=output_dim)
        self.cyber = CyberCore(input_dim=input_dim, output_dim=output_dim)
        self.goal = AngelGoalModule()
        self.sense = AngelSense(max_history=100, history_file="sense_history.pth")
        self.cogni = CogniCore()
        self.field_censor = TensorFieldCensor(output_dim=output_dim, num_classes=output_dim)
        
        self.stability = 0.0
        self.energy = 0.0
        self.cred = 0.0
        self.accuracy_history = []
        self.loss_history = []
        self.mood_history = []
        self.stability_history = []
        
        self.optimizer = None
        self.layer_scale = 1.0
        self.training_mode = True
        self.trainer_needed = False

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def set_training_mode(self, mode):
        self.training_mode = mode
        if not self.training_mode:
            self.eval()
            print("[AngelNet] Switched to autonomous classification mode")
        else:
            self.train()
            print("[AngelNet] Switched to supervised training mode")

    def adjust_learning_rate(self, resonance_factor):
        if self.optimizer is None or not self.training_mode:
            return
        lr_factor = 1.0 + resonance_factor * 0.5
        new_lr = self.base_lr * lr_factor
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
        print(f"[AngelNet] Adjusted learning rate: {new_lr:.6f} (factor: {lr_factor:.4f})")

    def adjust_layer_scale(self, curvature):
        if curvature > 0.5:
            self.layer_scale = 0.5
            print(f"[AngelNet] High curvature ({curvature:.4f}), reduced fc2 scale to {self.layer_scale:.4f}")
        else:
            self.layer_scale = 1.0
            print(f"[AngelNet] Normal curvature ({curvature:.4f}), restored fc2 scale to {self.layer_scale:.4f}")
    def forward(self, x, labels=None, data_type='image'):
        x = x.view(-1, self.input_dim)
        noise = torch.randn_like(x) * 0.3
        x = x + noise
        
        x = self.transformer.transform(x, data_type)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x)) * self.layer_scale
        x = self.fc3(x)
        
        if labels is not None and len(labels) > 0:
            field = self.vector_map.get_field(label=labels[0].item())
        else:
            field = self.vector_map.get_field()
        field = field.unsqueeze(0).expand(x.size(0), -1)
        x = x + field * 0.1
        
        return x

    def autonomous_classify(self, x, data_type='image'):
        output = self.forward(x, data_type=data_type)
        batch_size = output.size(0)
        predictions = torch.zeros(batch_size, dtype=torch.long, device=output.device)
        
        for i in range(batch_size):
            vector = output[i]
            predicted_class = self.vector_map.classify(vector)
            predictions[i] = predicted_class
        
        return predictions

    def decide_direction(self, vector, data_type, curvature, resonance_factor, mood_norm, interpretation, class_fields, label, accuracy):
        self.field_censor.update(class_fields, label, curvature, resonance_factor, accuracy, mood_norm)
        
        # Update class_fields in vector_map with gravity-adjusted fields
        self.vector_map.class_fields = self.field_censor.field_history[-1]['fields']
        
        is_familiar = self.vector_map.is_familiar(vector, data_type)
        entropy = interpretation['entropy']
        
        new_direction = self.field_censor.suggest_new_direction(class_fields, label, curvature, resonance_factor, accuracy, mood_norm)
        
        if not is_familiar or entropy > 1.0:
            print(f"[AngelNet] Vector is unfamiliar or uncertain (entropy={entropy:.4f}), enabling trainer")
            self.trainer_needed = True
            return True, new_direction
        
        self.trainer_needed = False
        
        if curvature > 0.5:
            print("[AngelNet] High curvature, slowing down updates")
        if resonance_factor > 0.5:
            print("[AngelNet] High resonance, accelerating updates")
        if abs(mood_norm - self.cogni.target_mood) > 0.5:
            print(f"[AngelNet] Mood deviation ({mood_norm:.4f} vs {self.cogni.target_mood:.4f}), adjusting behavior")
        
        return False, new_direction
    def think(self, x, labels, accuracy, success, loss, data_type='image'):
        output = self.forward(x, labels, data_type)
        self.cyber.update(x, output)
        
        vector = output[0]
        label = labels[0].item() if len(labels) > 0 else 0
        
        interpretation = self.vector_map.interpret_vector(vector, data_type)
        
        local_curvature = self.movie.compute_curvature(output)
        self.movie.update(output, local_curvature)
        resonance_vector = self.vector_map.get_resonance_vector()
        resonance_factor = self.vector_map.get_resonance_factor()
        
        self.goal.update(accuracy, success)
        reward = self.goal.get_reward()
        self.cogni.update(reward, success)
        mood_norm = self.cogni.mood_norm
        
        class_fields = self.vector_map.class_fields
        trainer_needed, new_direction = self.decide_direction(
            vector, data_type, local_curvature, resonance_factor, mood_norm, interpretation, class_fields, label, accuracy
        )
        
        if trainer_needed:
            self.set_training_mode(True)
        else:
            self.set_training_mode(False)
        
        if self.training_mode:
            probs = F.softmax(output, dim=1)
            max_probs, predicted = torch.max(probs, dim=1)
            confidence = max_probs.mean().item()
            self.vector_map.update(
                output, 
                label=label, 
                data_type=data_type, 
                confidence=confidence, 
                curvature=local_curvature, 
                resonance_factor=resonance_factor
            )
        else:
            self.vector_map.cluster_vectors(threshold=0.1)
        
        if new_direction is not None:
            print("[AngelNet] Applying new direction to fields")
            for i in range(self.num_classes):
                self.vector_map.class_fields[i] += new_direction * 0.1
        
        field = self.vector_map.get_field(label=label)
        self.global_field.update(field, local_curvature, resonance_factor)
        self.sense.sense(field, local_curvature, resonance_factor)
        self.intention.update(output, success, resonance_factor=resonance_factor)
        direction = self.intention.get_direction()
        self.action.update(direction, success)
        self.reflection.update(output, success)
        self.graph.update(output, direction)
        self.stability = self.compute_stability()
        self.energy = self.compute_energy()
        self.cred = self.compute_credibility()
        self.accuracy_history.append(accuracy)
        self.loss_history.append(loss)
        self.mood_history.append(self.cogni.mood_norm)
        self.stability_history.append(self.stability)
        
        # Print summary of AngelSense and TensorGlobalVectorMap
        self.sense.summary()
        self.vector_map.summary()

    def compute_stability(self):
        if len(self.accuracy_history) < 2:
            return 0.0
        recent_accuracies = self.accuracy_history[-10:]
        if len(recent_accuracies) < 2:
            return 0.0
        mean_acc = sum(recent_accuracies) / len(recent_accuracies)
        variance = sum((acc - mean_acc) ** 2 for acc in recent_accuracies) / len(recent_accuracies)
        stability = 1.0 / (1.0 + variance)
        return stability

    def compute_energy(self):
        if len(self.loss_history) == 0:
            return 0.0
        recent_losses = self.loss_history[-10:]
        if not recent_losses:
            return 0.0
        avg_loss = sum(recent_losses) / len(recent_losses)
        energy = 1.0 / (1.0 + avg_loss)
        return energy

    def compute_credibility(self):
        if len(self.accuracy_history) == 0:
            return 0.0
        recent_accuracies = self.accuracy_history[-10:]
        if not recent_accuracies:
            return 0.0
        avg_accuracy = sum(recent_accuracies) / len(recent_accuracies)
        return avg_accuracy

    def save_memory(self):
        self.vector_map.save_memory()
        self.vector_map.save_vector_history()
        self.transformer.save_segment('image')
        self.sense.save_history()

    def visualize_metrics(self, epoch):
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.plot(self.accuracy_history, label='Accuracy')
        plt.title('Accuracy Over Time')
        plt.xlabel('Batch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.subplot(2, 2, 2)
        plt.plot(self.loss_history, label='Loss', color='orange')
        plt.title('Loss Over Time')
        plt.xlabel('Batch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(2, 2, 3)
        plt.plot(self.mood_history, label='Mood', color='green')
        plt.title('Mood Over Time')
        plt.xlabel('Batch')
        plt.ylabel('Mood')
        plt.legend()
        
        plt.subplot(2, 2, 4)
        plt.plot(self.stability_history, label='Stability', color='purple')
        plt.title('Stability Over Time')
        plt.xlabel('Batch')
        plt.ylabel('Stability')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f'metrics_epoch_{epoch}.png')
        plt.close()    