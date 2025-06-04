import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical, Normal
import numpy as np

import gym
from tqdm import tqdm_notebook
from collections import deque


class Actor(nn.Module):
    
    def __init__(self, input_size, hidden_size, output_size):
        super(Actor, self).__init__()
        # TODO: IMPLEMENT ACTOR NETWORK ARCHITECTURE
        # Define network layers for actor policy
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.hidden_layer = nn.Linear(hidden_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)
    
    def forward(self, state):
        # TODO: IMPLEMENT FORWARD PASS
        # Process state through network layers with appropriate activations
        # Return action values in the proper range (e.g., using tanh activation)
        
        # Placeholder implementation
        return torch.zeros(state.shape[0], self.output_layer.out_features)


class Critic(nn.Module):
    def __init__(self, input_size,hidden_size, output_size):
        super(Critic, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)

    def forward(self, state, action):
        """
        Params state and actions are torch tensors
        """
        # IMPLEMENTATION REMOVED: Critic network forward pass
        return torch.zeros(state.shape[0], 1)
    
    