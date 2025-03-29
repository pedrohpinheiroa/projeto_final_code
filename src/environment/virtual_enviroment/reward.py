"""
Reward functions for the seesaw (gangorra) reinforcement learning environment.
This module implements different reward strategies to guide the learning process.
"""

import numpy as np
from typing import Tuple, Dict, Any, Callable


class RewardFunction:
    """
    Base class for reward functions used in the gangorra environment.
    
    Reward functions determine how the agent is incentivized to learn
    the desired behavior of balancing the seesaw.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the reward function with the given parameters.
        
        Args:
            config: Dictionary containing parameters for the reward function.
                   If None, default values will be used.
        """
        # Default configuration
        default_config = {
            "angle_weight": 1.0,        # Weight for angle component
            "velocity_weight": 0.5,      # Weight for angular velocity component
            "action_weight": 0.1,        # Weight for action effort component
            "time_weight": 0.01,         # Weight for time component
            "balance_reward": 10.0,      # Reward for achieving balance
            "balance_tolerance": 0.01,   # Tolerance for considering balanced state
            "early_termination": True,   # Terminate episode when balanced
            "termination_steps": 50,     # Steps to maintain balance for termination
        }
        
        # Update default config with provided config
        self.config = default_config.copy()
        if config is not None:
            self.config.update(config)
            
        # Counter for steps in balanced state
        self.balanced_steps = 0
    
    def reset(self) -> None:
        """Reset the internal state of the reward function."""
        self.balanced_steps = 0
    
    def __call__(self, 
                state: Tuple[float, float], 
                action: Tuple[float, float],
                next_state: Tuple[float, float],
                time_step: float) -> Tuple[float, bool]:
        """
        Calculate the reward for a transition.
        
        Args:
            state: Current state (angle, angular_velocity)
            action: Action taken (left_fan_force, right_fan_force)
            next_state: Resulting state (angle, angular_velocity)
            time_step: Current time step of the episode
            
        Returns:
            Tuple of (reward, done) where done indicates if the episode
            should be terminated.
        """
        raise NotImplementedError("Subclasses must implement __call__ method")


class BasicReward(RewardFunction):
    """
    Basic reward function that penalizes deviation from the balanced state
    and rewards stability.
    """
    
    def __call__(self, 
                state: Tuple[float, float], 
                action: Tuple[float, float],
                next_state: Tuple[float, float],
                time_step: float) -> Tuple[float, bool]:
        """
        Calculate reward based on how close the seesaw is to being balanced.
        
        Args:
            state: Current state (angle, angular_velocity)
            action: Action taken (left_fan_force, right_fan_force)
            next_state: Resulting state (angle, angular_velocity)
            time_step: Current time step of the episode
            
        Returns:
            Tuple of (reward, done) where done indicates if the episode
            should be terminated.
        """
        next_angle, next_angular_velocity = next_state
        left_force, right_force = action
        
        # Angle component: penalize deviation from 0
        angle_reward = -self.config["angle_weight"] * abs(next_angle)
        
        # Velocity component: penalize high velocities
        velocity_reward = -self.config["velocity_weight"] * abs(next_angular_velocity)
        
        # Action component: penalize high action values (energy efficiency)
        action_magnitude = abs(left_force) + abs(right_force)
        action_reward = -self.config["action_weight"] * action_magnitude
        
        # Time component: slight penalty for each time step (encourages faster solving)
        time_reward = -self.config["time_weight"]
        
        # Total reward
        reward = angle_reward + velocity_reward + action_reward + time_reward
        
        # Check if balanced
        is_balanced = (abs(next_angle) < self.config["balance_tolerance"] and 
                       abs(next_angular_velocity) < self.config["balance_tolerance"])
        
        # Update balanced steps counter
        if is_balanced:
            self.balanced_steps += 1
            # Add bonus reward for being balanced
            reward += self.config["balance_reward"]
        else:
            self.balanced_steps = 0
        
        # Determine if episode is done
        done = False
        if (self.config["early_termination"] and 
            self.balanced_steps >= self.config["termination_steps"]):
            done = True
        
        return reward, done


class AsymptoticalReward(RewardFunction):
    """
    Asymptotical reward function that provides smoother gradient
    for learning and more aggressively rewards close-to-balance states.
    """
    
    def __call__(self, 
                state: Tuple[float, float], 
                action: Tuple[float, float],
                next_state: Tuple[float, float],
                time_step: float) -> Tuple[float, bool]:
        """
        Calculate reward using asymptotic functions for smoother gradients.
        
        Args:
            state: Current state (angle, angular_velocity)
            action: Action taken (left_fan_force, right_fan_force)
            next_state: Resulting state (angle, angular_velocity)
            time_step: Current time step of the episode
            
        Returns:
            Tuple of (reward, done) where done indicates if the episode
            should be terminated.
        """
        next_angle, next_angular_velocity = next_state
        left_force, right_force = action
        
        # Angle component: exponential decay reward
        angle_reward = self.config["angle_weight"] * np.exp(-10 * abs(next_angle))
        
        # Velocity component: exponential decay reward
        velocity_reward = self.config["velocity_weight"] * np.exp(-5 * abs(next_angular_velocity))
        
        # Action component: penalize high action values (energy efficiency)
        action_magnitude = abs(left_force) + abs(right_force)
        action_reward = -self.config["action_weight"] * action_magnitude
        
        # Time component: slight penalty for each time step (encourages faster solving)
        time_reward = -self.config["time_weight"]
        
        # Total reward
        reward = angle_reward + velocity_reward + action_reward + time_reward
        
        # Check if balanced
        is_balanced = (abs(next_angle) < self.config["balance_tolerance"] and 
                       abs(next_angular_velocity) < self.config["balance_tolerance"])
        
        # Update balanced steps counter
        if is_balanced:
            self.balanced_steps += 1
            # Add bonus reward for being balanced
            reward += self.config["balance_reward"]
        else:
            self.balanced_steps = 0
        
        # Determine if episode is done
        done = False
        if (self.config["early_termination"] and 
            self.balanced_steps >= self.config["termination_steps"]):
            done = True
        
        return reward, done


class SparseReward(RewardFunction):
    """
    Sparse reward function that mainly rewards the agent for achieving balance
    and maintaining it, with minimal guidance otherwise.
    """
    
    def __call__(self, 
                state: Tuple[float, float], 
                action: Tuple[float, float],
                next_state: Tuple[float, float],
                time_step: float) -> Tuple[float, bool]:
        """
        Calculate sparse reward focused on the balanced state.
        
        Args:
            state: Current state (angle, angular_velocity)
            action: Action taken (left_fan_force, right_fan_force)
            next_state: Resulting state (angle, angular_velocity)
            time_step: Current time step of the episode
            
        Returns:
            Tuple of (reward, done) where done indicates if the episode
            should be terminated.
        """
        next_angle, next_angular_velocity = next_state
        
        # Small penalty for each time step (encourages faster solving)
        reward = -self.config["time_weight"]
        
        # Check if balanced
        is_balanced = (abs(next_angle) < self.config["balance_tolerance"] and 
                       abs(next_angular_velocity) < self.config["balance_tolerance"])
        
        # Update balanced steps counter and assign reward
        if is_balanced:
            self.balanced_steps += 1
            reward = self.config["balance_reward"]
        else:
            self.balanced_steps = 0
        
        # Determine if episode is done
        done = False
        if (self.config["early_termination"] and 
            self.balanced_steps >= self.config["termination_steps"]):
            done = True
        
        return reward, done


def get_reward_function(name: str, config: Dict[str, Any] = None) -> RewardFunction:
    """
    Factory function to get the appropriate reward function by name.
    
    Args:
        name: Name of the reward function to use.
        config: Configuration for the reward function.
    
    Returns:
        An instance of the specified reward function.
    
    Raises:
        ValueError: If the specified reward function is not found.
    """
    reward_functions = {
        "basic": BasicReward,
        "asymptotic": AsymptoticalReward,
        "sparse": SparseReward
    }
    
    if name not in reward_functions:
        raise ValueError(f"Unknown reward function: {name}. Available options: {list(reward_functions.keys())}")
    
    return reward_functions[name](config)