import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_stacked_layers,device):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_stacked_layers = num_stacked_layers
        self.device=device
        self.lstm = nn.LSTM(input_size, hidden_size, num_stacked_layers,
                            batch_first=True)
        self.act=nn.Sigmoid()
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(self.device)

        out, _ = self.lstm(x.float(), (h0, c0))
        out = self.act(self.fc(out[:, -1, :]))
        return out
    
    

class AttentionModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_heads):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads

        # Linear transformations for query, key, and value
        self.query_linear = nn.Linear(input_size, hidden_size)
        self.key_linear = nn.Linear(input_size, hidden_size)
        self.value_linear = nn.Linear(input_size, hidden_size)

        self.fc = nn.Linear(hidden_size, 1)
        self.act=nn.Sigmoid()


    def scaled_dot_product_attention(self, query, key, value):
        """
        Calculates scaled dot-product attention.

        Args:
            query (torch.Tensor): Query tensor (batch_size, seq_len, hidden_size).
            key (torch.Tensor): Key tensor (batch_size, seq_len, hidden_size).
            value (torch.Tensor): Value tensor (batch_size, seq_len, hidden_size).

        Returns:
            torch.Tensor: Attention output (batch_size, hidden_size).
        """
        d_k = query.size(-1)  # hidden_size

        # Calculate attention scores (batch_size, seq_len, seq_len)
        scores = torch.matmul(query, key.transpose(-2, -1)) / np.sqrt(d_k)

        # Apply softmax to get attention weights (batch_size, seq_len, seq_len)
        attention_weights = F.softmax(scores, dim=-1)

        # Apply attention weights to values (batch_size, seq_len, hidden_size)
        attention_output = torch.matmul(attention_weights, value)
        # Average the output over the sequence length (batch_size, hidden_size)
        attention_output = torch.mean(attention_output, dim=1)
        return attention_output

    def forward(self, x):
        """
        Forward pass of the attention model.

        Args:
            x (torch.Tensor): Input tensor (batch_size, seq_len, input_size).

        Returns:
            torch.Tensor: Output tensor (batch_size, 1).
        """
        batch_size = x.size(0)
        seq_len = x.size(1)

        # Linear transformations
        query = self.query_linear(x)  # (batch_size, seq_len, hidden_size)
        key = self.key_linear(x)  # (batch_size, seq_len, hidden_size)
        value = self.value_linear(x)  # (batch_size, seq_len, hidden_size)

        # Calculate attention output
        attention_output = self.scaled_dot_product_attention(query, key, value)  # (batch_size, hidden_size)

        # Fully connected layer and sigmoid activation
        out = (self.fc(attention_output))  # (batch_size, 1)
        return out


if __name__ == "__main__":
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    model = LSTM(4, 128, 3,device)
    model
    # Example usage:
    input_size = 4  # Number of features
    hidden_size = 128  # Hidden size for attention
    num_heads = 8 # Number of attention heads
    model = AttentionModel(input_size, hidden_size, num_heads)
    model.to(device)
    print(model)
