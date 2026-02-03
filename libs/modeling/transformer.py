import torch.nn as nn
from torch.nn import MultiheadAttention
    

class FeedForward(nn.Module):
    def __init__(self, embed_size, ff_ratio=4.0, dropout=0.1):
        """
        Args:
            embed_size (int): The size of the embedding.
            ff_ratio (float): The ratio to compute the hidden dimension of the feed-forward network.
            dropout (float): Dropout rate.
        """
        super(FeedForward, self).__init__()
        ff_hidden = int(embed_size * ff_ratio)
        self.fc1 = nn.Linear(embed_size, ff_hidden)
        self.fc2 = nn.Linear(ff_hidden, embed_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_size, num_heads, ff_ratio=4.0, dropout=0.1):
        """
        Args:
            embed_size (int): The size of the embedding.
            num_heads (int): Number of attention heads.
            ff_ratio (float): Ratio to compute the feed-forward hidden dimension.
            dropout (float): Dropout rate.
        """
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(embed_size, num_heads, dropout=dropout)
        self.feed_forward = FeedForward(embed_size, ff_ratio, dropout)
        
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, src, src_mask=None):
        """
        Args:
            src (Tensor): Input tensor of shape (batch_size, seq_length, embed_size).
            src_mask (Tensor, optional): Mask tensor of shape (batch_size, 1, 1, seq_length) or appropriate shape.
        
        Returns:
            Tensor: Output tensor of shape (batch_size, seq_length, embed_size).
        """
        # Self-Attention sublayer
        attn_output = self.self_attn(src, src, src, src_mask)  # (N, seq_length, embed_size)
        src = self.norm1(src + self.dropout(attn_output))      # Residual connection + LayerNorm
        
        # Feed-Forward sublayer
        ff_output = self.feed_forward(src)                     # (N, seq_length, embed_size)
        src = self.norm2(src + self.dropout(ff_output))        # Residual connection + LayerNorm
        
        return src
    

class TransformerEncoder(nn.Module):
    def __init__(self, 
                 embed_size, 
                 num_layers, 
                 num_heads=8, 
                 ff_ratio=4.0, 
                 dropout=0.1):
        """
        Args:
            embed_size (int): The size of the embedding.
            num_layers (int): Number of encoder layers.
            num_heads (int): Number of attention heads.
            ff_ratio (float): Ratio to compute the feed-forward hidden dimension.
            dropout (float): Dropout rate.
        """
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(embed_size, num_heads, ff_ratio, dropout)
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_size)
    
    def forward(self, src, src_mask=None):
        """
        Args:
            src (Tensor): Input tensor of shape (batch_size, seq_length, embed_size).
            src_mask (Tensor, optional): Mask tensor.
        
        Returns:
            Tensor: Output tensor of shape (batch_size, seq_length, embed_size).
        """
        src = self.dropout(src)
        for layer in self.layers:
            src = layer(src, src_mask)
        src = self.norm(src)
        return src
    

class TransformerDecoderLayer(nn.Module):
    def __init__(self, embed_size, num_heads, ff_ratio=4.0, dropout=0.1):
        """
        Args:
            embed_size (int): The size of the embedding.
            num_heads (int): Number of attention heads.
            ff_ratio (float): Ratio to compute the feed-forward hidden dimension.
            dropout (float): Dropout rate.
        """
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(embed_size, num_heads, dropout)
        self.src_attn = MultiheadAttention(embed_size, num_heads, dropout)
        self.feed_forward = FeedForward(embed_size, ff_ratio, dropout)
        
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.norm3 = nn.LayerNorm(embed_size)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        """
        Args:
            tgt (Tensor): Input tensor of shape (batch_size, tgt_length, embed_size).
            memory (Tensor): Input tensor from the encoder of shape (batch_size, src_length, embed_size).
            tgt_mask (Tensor, optional): Mask tensor for target sequence.
            memory_mask (Tensor, optional): Mask tensor for source sequence.
        
        Returns:
            Tensor: Output tensor of shape (batch_size, tgt_length, embed_size).
        """
        # Self-Attention sublayer
        attn_output = self.self_attn(tgt, tgt, tgt, tgt_mask)  # (N, tgt_length, embed_size)
        tgt = self.norm1(tgt + self.dropout(attn_output))      # Residual connection + LayerNorm
        
        # Source-Attention sublayer
        attn_output = self.src_attn(tgt, memory, memory, memory_mask)  # (N, tgt_length, embed_size)
        tgt = self.norm2(tgt + self.dropout(attn_output))               # Residual connection + LayerNorm
        
        # Feed-Forward sublayer
        ff_output = self.feed_forward(tgt)                     # (N, tgt_length, embed_size)
        tgt = self.norm3(tgt + self.dropout(ff_output))        # Residual connection + LayerNorm
        
        return tgt
    

class TransformerDecoder(nn.Module):
    def __init__(self, 
                 embed_size, 
                 num_layers, 
                 num_heads=8, 
                 ff_ratio=4.0, 
                 dropout=0.1):
        """
        Args:
            embed_size (int): The size of the embedding.
            num_layers (int): Number of decoder layers.
            num_heads (int): Number of attention heads.
            ff_ratio (float): Ratio to compute the feed-forward hidden dimension.
            dropout (float): Dropout rate.
        """
        super(TransformerDecoder, self).__init__()
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(embed_size, num_heads, ff_ratio, dropout)
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_size)
    
    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        """
        Args:
            tgt (Tensor): Input tensor of shape (batch_size, tgt_length, embed_size).
            memory (Tensor): Input tensor from the encoder of shape (batch_size, src_length, embed_size).
            tgt_mask (Tensor, optional): Mask tensor for target sequence.
            memory_mask (Tensor, optional): Mask tensor for source sequence.
        
        Returns:
            Tensor: Output tensor of shape (batch_size, tgt_length, embed_size).
        """
        tgt = self.dropout(tgt)
        for layer in self.layers:
            tgt = layer(tgt, memory, tgt_mask, memory_mask)
        tgt = self.norm(tgt)
        return tgt