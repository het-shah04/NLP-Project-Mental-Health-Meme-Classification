# 
import torch
import torch.nn as nn

class OCRAttention(nn.Module):
    def __init__(self, config):
        super(OCRAttention, self).__init__()
        self.config = config
        
        # Initialize the linear layers
        self.K = nn.Linear(config['hidden_size'], config['hidden_size'])  # Linear projection for K (keys)
        self.Q = nn.Linear(config['hidden_size'], config['hidden_size'])  # Linear projection for Q (query)
        self.V = nn.Linear(config['hidden_size'], config['hidden_size'])  # Linear projection for V (values)

    def forward(self, current_triple, k_triples):
        """
        current_triple: Tensor of shape [batch_size, hidden_size] -> query
        k_triples: Tensor of shape [batch_size, k, hidden_size] -> keys and values

        Returns:
            output: Tensor of shape [batch_size, hidden_size] -> attention representation
        """
        
        # Step 1: Apply linear layers to get Q, K, V
        query = self.Q(current_triple)  # [batch_size, hidden_size]
        keys = self.K(k_triples)  # [batch_size, k, hidden_size]
        values = self.V(k_triples)  # [batch_size, k, hidden_size]

        # Step 2: Compute the attention scores (dot product between Q and K)
        # Query: [batch_size, hidden_size] -> expand to [batch_size, 1, hidden_size]
        query = query.unsqueeze(1)  # [batch_size, 1, hidden_size]
        
        # Compute the attention scores: [batch_size, 1, hidden_size] x [batch_size, k, hidden_size]^T -> [batch_size, 1, k]
        attention_scores = torch.bmm(query, keys.transpose(1, 2))  # [batch_size, 1, k]
        
        # Step 3: Apply softmax to get the attention weights
        attention_weights = torch.softmax(attention_scores, dim=-1)  # [batch_size, 1, k]
        
        # Step 4: Compute the attention output as the weighted sum of values
        # attention_weights: [batch_size, 1, k] x values: [batch_size, k, hidden_size] -> [batch_size, 1, hidden_size]
        attention_output = torch.bmm(attention_weights, values)  # [batch_size, 1, hidden_size]

        # Step 5: Squeeze the extra dimensions and return the final attention representation
        output = attention_output.squeeze(1)  # [batch_size, hidden_size]

        return output



