# cogni_core.py
import torch

class CogniCore:
    """
    Manages the cognitive state and mood of the system.
    """
    def __init__(self):
        self.mood_norm = 0.0
        self.target_mood = 0.6
        self.mood_components = [0.0, 0.0, 0.0]
        self.steps_unstable = 0
    
    def update(self, reward, success):
        """
        Update the mood based on reward and success.
        
        Args:
            reward (float): Reward value.
            success (bool): Whether the goal was achieved.
        """
        # Обновляем настроение
        mood_change = reward * 0.05 if success else -0.02  # Уменьшили скорость изменения
        self.mood_norm += mood_change
        # Ограничиваем Mood Norm
        self.mood_norm = max(0.0, min(self.mood_norm, 5.0))
        # Обновляем компоненты настроения (заглушка)
        self.mood_components = [torch.randn(1).item(), torch.randn(1).item() * 2, torch.randn(1).item()]
        # Проверяем отклонение
        deviation = abs(self.mood_norm - self.target_mood)
        if deviation > 0.1:
            self.steps_unstable += 1
        else:
            self.steps_unstable = 0
        print(f"[CogniCore] Mood Components: {self.mood_components}")
        print(f"Mood (norm): +{self.mood_norm:.4f}, Target: {self.target_mood:.4f}, Reaction: {mood_change:.4f}")
        print(f"Deviation: {deviation:.4f}, Steps Unstable: {self.steps_unstable}")
    
    def summary(self):
        """
        Print a summary of the cognitive state.
        """
        print(f"[CogniCore] Mood Norm: {self.mood_norm:.4f}, Target Mood: {self.target_mood:.4f}")