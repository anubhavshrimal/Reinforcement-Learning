import torch
import torch.nn as nn
import torch.nn.functional as F

#############################
# Dueling DQN Architecture  #
#############################
class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, action_size, in_channels=1, seed=42):
        """Initialize parameters and build model.
        Params
        ======
            action_size (int): Dimension of each action
            in_channels (int): Number of input channels for each pixel state / image
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.action_size = action_size
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        self.state_fc = nn.Sequential(
            nn.Linear(in_features=7*7*64, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=1)
        )
        
        self.advantage_fc = nn.Sequential(
            nn.Linear(in_features=7*7*64, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=action_size)
        )
        

    def forward(self, state):
        """Build a network that maps state -> action values."""
        out = self.conv(state)
        out = out.view(state.size(0), -1)
        
        state_value = self.state_fc(out)
        advantage_value = self.advantage_fc(out)
        
        state_value = state_value.expand(state.size(0), self.action_size)
        
        out = state_value + advantage_value - advantage_value.mean(1).unsqueeze(1).expand(state.size(0), self.action_size)
        
        return out
