import torch
import torch.optim as optim
import torch.nn as nn
from model import SkeletonModel

class AIBrain:
    """
    The 'Brain' of the AI, responsible for processing information 
    and learning from experience (Self-Learning).
    """
    def __init__(self, model: SkeletonModel, lr=0.001):
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss() # Default to Mean Squared Error
        
    def think(self, data):
        """
        Processes input data and returns a prediction.
        """
        self.model.eval()
        with torch.no_grad():
            tensor_data = torch.FloatTensor(data)
            prediction = self.model(tensor_data)
        return prediction.numpy()

    def learn(self, input_data, target_data):
        """
        Updates the model's weights based on the error between 
        prediction and target (The Learning Step).
        """
        self.model.train()
        
        # Convert to tensors
        inputs = torch.FloatTensor(input_data)
        targets = torch.FloatTensor(target_data)
        
        # Zero the gradients
        self.optimizer.zero_grad()
        
        # Forward pass
        outputs = self.model(inputs)
        
        # Calculate Loss
        loss = self.criterion(outputs, targets)
        
        # Backward pass (Calculate gradients)
        loss.backward()
        
        # Update weights
        self.optimizer.step()
        
        return loss.item()

    def save_knowledge(self, path="knowledge.pth"):
        """Saves the learned weights to a file."""
        torch.save(self.model.state_dict(), path)

    def load_knowledge(self, path="knowledge.pth"):
        """Loads weights from a file."""
        self.model.load_state_dict(torch.load(path))
        self.model.eval()
