"""
Attention mechanism utilities for interpretability.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict


class InterpretableMultiHeadAttention(nn.Module):
    """
    Interpretable Multi-Head Attention from TFT.
    
    Unlike standard multi-head attention, this variant:
    - Shares the same query, key, value projections across heads
    - Averages attention weights across heads for interpretability
    - Maintains single set of output projections
    """
    
    def __init__(
        self,
        hidden_size: int,
        head_count: int,
        dropout: float = 0.1,
        eps: float = 1e-6
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.head_count = head_count
        self.head_dim = hidden_size // head_count
        self.eps = eps
        
        assert hidden_size % head_count == 0, \
            f"hidden_size ({hidden_size}) must be divisible by head_count ({head_count})"
        
        # Shared projections for Q, K, V across all heads
        self.qkv_projection = nn.Linear(hidden_size, 3 * hidden_size)
        
        # Output projection
        self.out_projection = nn.Linear(hidden_size, hidden_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Scaling factor
        self.scale = self.head_dim ** -0.5
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            query: [batch, query_len, hidden_size]
            key: [batch, key_len, hidden_size]
            value: [batch, key_len, hidden_size]
            mask: [batch, key_len] or [batch, query_len, key_len]
            
        Returns:
            output: [batch, query_len, hidden_size]
            attention_weights: [batch, query_len, key_len] (averaged over heads)
        """
        batch_size, query_len, _ = query.shape
        key_len = key.shape[1]
        
        # Project Q, K, V
        qkv = self.qkv_projection(query)  # For query input
        # Note: In TFT, same projection for encoder/decoder
        q = qkv[:, :, :self.hidden_size]
        
        # For key/value, use their own projections
        kv = self.qkv_projection(key)
        k = kv[:, :, :self.hidden_size]
        v = kv[:, :, self.hidden_size:2*self.hidden_size]
        
        # Reshape for multi-head: [B, L, H] -> [B, heads, L, head_dim]
        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, heads, Q, K]
        
        # Apply mask
        if mask is not None:
            if mask.dim() == 2:
                # [B, K] -> [B, 1, 1, K]
                mask = mask.unsqueeze(1).unsqueeze(2)
            elif mask.dim() == 3:
                # [B, Q, K] -> [B, 1, Q, K]
                mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Attention weights
        attn_weights = F.softmax(scores, dim=-1)  # [B, heads, Q, K]
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        context = torch.matmul(attn_weights, v)  # [B, heads, Q, head_dim]
        
        # Concatenate heads
        context = self._merge_heads(context)  # [B, Q, H]
        
        # Output projection
        output = self.out_projection(context)
        
        # Average attention weights over heads for interpretability
        avg_attn = attn_weights.mean(dim=1)  # [B, Q, K]
        
        return output, avg_attn
    
    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Split last dimension into multiple heads."""
        batch_size, seq_len, _ = x.shape
        x = x.view(batch_size, seq_len, self.head_count, self.head_dim)
        return x.transpose(1, 2)  # [B, heads, L, head_dim]
    
    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Merge multiple heads back to single dimension."""
        batch_size, _, seq_len, _ = x.shape
        x = x.transpose(1, 2).contiguous()  # [B, L, heads, head_dim]
        return x.view(batch_size, seq_len, self.hidden_size)


class VariableSelectionNetwork(nn.Module):
    """
    Variable Selection Network from TFT.
    
    Learns to weight input features based on their relevance
    using a gating mechanism and softmax normalization.
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        dropout: float = 0.1,
        gate_additive: bool = True,
        gate_multiplier: float = 2.0
    ):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.gate_additive = gate_additive
        self.gate_multiplier = gate_multiplier
        
        # Feature transformation
        self.feature_transform = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, hidden_size)
        )
        
        # Variable selection weights
        self.selector = nn.Linear(hidden_size, input_size)
        
        # Gating mechanism
        self.gate = nn.Linear(input_size, hidden_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Layer norm
        self.layer_norm = nn.LayerNorm(hidden_size)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input features [batch, time, input_size] or [batch, input_size]
            mask: Optional mask [batch, time] for temporal inputs
            
        Returns:
            output: Selected/weighted features [batch, time, hidden_size] or [batch, hidden_size]
            weights: Variable selection weights [batch, time, input_size] or [batch, input_size]
        """
        is_temporal = x.dim() == 3
        
        if is_temporal:
            batch_size, time_steps, _ = x.shape
            # Flatten time dimension for processing
            x_flat = x.view(-1, self.input_size)  # [B*T, input_size]
        else:
            x_flat = x
        
        # Transform features
        transformed = self.feature_transform(x_flat)  # [B*T or B, hidden_size]
        
        # Compute selection weights
        selection_logits = self.selector(transformed)  # [B*T or B, input_size]
        
        # Apply mask if provided
        if mask is not None and is_temporal:
            mask_flat = mask.view(-1, 1)  # [B*T, 1]
            selection_logits = selection_logits.masked_fill(mask_flat == 0, -1e9)
        
        # Softmax over input features
        weights = F.softmax(selection_logits, dim=-1)  # [B*T or B, input_size]
        weights = self.dropout(weights)
        
        # Apply weights to original input
        weighted_input = torch.sum(
            weights.unsqueeze(-1) * x_flat.unsqueeze(-2),  # [B*T, 1, input_size] * [B*T, input_size, 1]
            dim=-1
        )  # [B*T, input_size]
        
        # Gate mechanism
        gate_input = x_flat if self.gate_additive else weighted_input
        gate = torch.sigmoid(self.gate(gate_input) * self.gate_multiplier)
        
        # Apply gate
        if self.gate_additive:
            output = weighted_input * gate
        else:
            output = weighted_input
        
        # Layer norm
        output = self.layer_norm(output)
        
        # Reshape if temporal
        if is_temporal:
            output = output.view(batch_size, time_steps, self.hidden_size)
            weights = weights.view(batch_size, time_steps, self.input_size)
        
        return output, weights


def extract_attention_weights(
    model_output: Dict[str, torch.Tensor],
    token_ids: Optional[torch.Tensor] = None,
    aggregate_heads: bool = True
) -> Dict[str, torch.Tensor]:
    """
    Extract and process attention weights from model output.
    
    Args:
        model_output: Output dictionary from TFT forward pass
        token_ids: Optional token IDs for mapping attention to words
        aggregate_heads: Whether to average over attention heads
        
    Returns:
        Dictionary with processed attention weights
    """
    result = {}
    
    if 'attention_weights' in model_output:
        attn = model_output['attention_weights']  # [B, heads, Q, K] or [B, Q, K]
        
        if aggregate_heads and attn.dim() == 4:
            attn = attn.mean(dim=1)  # [B, Q, K]
        
        result['temporal_attention'] = attn
    
    if 'variable_selection_weights' in model_output:
        result['variable_selection'] = model_output['variable_selection_weights']
    
    return result


def compute_word_impact(
    finbert_attention: torch.Tensor,      # [batch, seq_len] - token attention
    vsn_weights: torch.Tensor,             # [batch, input_size] - news feature weight
    predicted_direction: torch.Tensor,     # [batch] or scalar - +1/-1
    news_to_token_map: Dict[int, list],    # Maps news index to token indices
    normalize: bool = True
) -> torch.Tensor:
    """
    Compute word-level impact scores for criteria extraction.
    
    Formula: Impact(w_i) = α_FinBERT(w_i) × β_VSN(N_k) × sgn(ŷ_t)
    
    Args:
        finbert_attention: Token-level attention from FinBERT
        vsn_weights: Variable selection weights for news items
        predicted_direction: Predicted return direction (+1 or -1)
        news_to_token_map: Mapping from news item index to token positions
        normalize: Whether to normalize impact scores
        
    Returns:
        Word impact scores [batch, seq_len]
    """
    # Ensure direction is scalar or [batch]
    if predicted_direction.dim() == 0:
        direction = predicted_direction
    else:
        direction = predicted_direction.unsqueeze(-1)  # [batch, 1]
    
    # Get news item weights for each token
    token_news_weights = torch.zeros_like(finbert_attention)
    
    for news_idx, token_indices in news_to_token_map.items():
        if news_idx < vsn_weights.shape[-1]:
            news_weight = vsn_weights[..., news_idx]  # [batch] or scalar
            for token_idx in token_indices:
                if token_idx < token_news_weights.shape[-1]:
                    token_news_weights[..., token_idx] = news_weight
    
    # Compute impact
    impact = finbert_attention * token_news_weights * direction
    
    # Normalize if requested
    if normalize:
        impact = impact / (impact.abs().sum(dim=-1, keepdim=True) + 1e-9)
    
    return impact
