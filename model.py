import torch
import torch.nn as nn
import torch.nn.functional as F

class SkeletonModel(nn.Module):
    """
    A skeleton neural network model with no prior knowledge.
    It is randomly initialized and ready to learn.
    """
    def __init__(self, input_size=10, hidden_size=64, output_size=2):
        super(SkeletonModel, self).__init__()
        # Input layer to hidden layer
        self.fc1 = nn.Linear(input_size, hidden_size)
        # Hidden layer to hidden layer
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        # Hidden layer to output layer
        self.fc3 = nn.Linear(hidden_size, output_size)
        
        # Initialize weights randomly (this ensures the model starts with no knowledge)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Forward pass through the network.
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
