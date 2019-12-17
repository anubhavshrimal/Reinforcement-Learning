import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, hidden_dims=[64, 32]):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        self.fc_layers = nn.ModuleList()

        # define the model architecture
        current_dim = state_size
        for hdim in hidden_dims:
            self.fc_layers.append(nn.Linear(current_dim, hdim))
            current_dim = hdim
        self.fc_layers.append(nn.Linear(current_dim, action_size))
        

    def forward(self, state):
        """Build a network that maps state -> action values."""
        out = state
        
        for layer in self.fc_layers[:-1]:
            out = F.relu(layer(out))
            
        out = self.fc_layers[-1](out)
        return out
