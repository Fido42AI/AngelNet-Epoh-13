# angel_goal_module.py
# -*- coding: utf-8 -*-
# AngelGoalModule for AngelNet
# Author: Fedorenko Bohdan (Angel42)
# Updated by: Grok CPT-Chat
# Version: 13.7.1 (Added initialization)

class AngelGoalModule:
    """
    AngelGoalModule for managing goals and tracking performance.
    """
    def __init__(self, target_accuracy=0.9):
        """
        Initialize the AngelGoalModule.
        
        Args:
            target_accuracy (float): Target accuracy for the system.
        """
        self.target_accuracy = target_accuracy
        self.current_accuracy = 0.0
        self.current_loss = 0.0
        self.reward = 0.0
    
    def update(self, accuracy, success, loss=None):
        """
        Update the goal module with the current accuracy and success.

        Args:
            accuracy (float): Current accuracy.
            success (bool): Whether the goal was achieved.
            loss (float, optional): Latest loss value from the learner.
        """
        self.current_accuracy = accuracy
        if loss is not None:
            self.current_loss = float(loss)
        else:
            # Fall back to a simple proxy that rewards higher accuracy.
            self.current_loss = float(1.0 - accuracy)
        self.reward = 1.0 if success else -0.5

        print(f"[AngelGoalModule] Updated - Accuracy: {self.current_accuracy:.4f}, Loss: {self.current_loss:.4f}")
        if self.current_accuracy >= self.target_accuracy:
            print(f"[AngelGoalModule] Target accuracy {self.target_accuracy} reached!")
    
    def get_reward(self):
        """
        Get the current reward.
        
        Returns:
            float: Reward value.
        """
        return self.reward
    
    def summary(self):
        """
        Print a summary of the goal module state.
        """
        print(f"[AngelGoalModule] Target: {self.target_accuracy}, Current Accuracy: {self.current_accuracy:.4f}, Current Loss: {self.current_loss:.4f}")