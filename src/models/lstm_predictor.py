import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional

class LSTMPredictor(nn.Module):
    """LSTM-based access pattern predictor.
    
    Predicts future access probabilities based on recent access history.
    """
    
    def __init__(self, 
                 input_size: int,
                 hidden_size: int = 128,
                 num_layers: int = 2,
                 dropout: float = 0.1):
        """Initialize LSTM predictor.
        
        Args:
            input_size: Size of input features (e.g., one-hot block ID)
            hidden_size: Number of LSTM hidden units
            num_layers: Number of LSTM layers
            dropout: Dropout probability
        """
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, input_size),
            nn.Sigmoid()
        )
        
    def forward(self, 
                x: torch.Tensor,
                hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
               ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
            hidden: Optional initial hidden state
            
        Returns:
            (predictions, hidden_state)
            predictions: Access probabilities for next timestep
            hidden_state: Updated LSTM hidden state
        """
        batch_size = x.size(0)
        
        if hidden is None:
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size,
                           device=x.device)
            c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size,
                           device=x.device)
            hidden = (h0, c0)
            
        # LSTM forward pass
        lstm_out, hidden = self.lstm(x, hidden)
        
        # Predict next access probabilities
        predictions = self.fc(lstm_out[:, -1, :])
        
        return predictions, hidden
    
    def init_hidden(self, batch_size: int, device: torch.device
                  ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize hidden state."""
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size,
                        device=device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size,
                        device=device)
        return (h0, c0)
    
class AccessHistoryBuffer:
    """Buffer for maintaining access history for LSTM prediction."""
    
    def __init__(self,
                 num_blocks: int,
                 history_length: int = 32,
                 device: torch.device = None):
        """Initialize access history buffer.
        
        Args:
            num_blocks: Total number of cache blocks
            history_length: Number of past accesses to track
            device: Torch device
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
        self.num_blocks = num_blocks
        self.history_length = history_length
        self.device = device
        
        # Initialize circular buffer
        self.history = torch.zeros(
            (1, history_length, num_blocks),
            dtype=torch.float32,
            device=device
        )
        self.position = 0
        
    def add_access(self, block_id: int):
        """Record a block access."""
        # Create one-hot encoding
        access = torch.zeros(1, 1, self.num_blocks, device=self.device)
        access[0, 0, block_id] = 1.0
        
        # Insert at current position
        self.history[:, self.position:self.position+1, :] = access
        
        # Update position
        self.position = (self.position + 1) % self.history_length
        
    def get_history(self) -> torch.Tensor:
        """Get recent access history.
        
        Returns a tensor of shape (1, history_length, num_blocks)
        with accesses ordered from oldest to newest.
        """
        # Reorder buffer so oldest access is first
        if self.position == 0:
            return self.history
        
        history = torch.cat([
            self.history[:, self.position:, :],
            self.history[:, :self.position, :]
        ], dim=1)
        
        return history
    
    def clear(self):
        """Reset access history."""
        self.history.zero_()
        self.position = 0