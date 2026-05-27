"""
Temporal Fusion Transformer implementation for interpretable forecasting.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
from loguru import logger

from .attention_utils import InterpretableMultiHeadAttention, VariableSelectionNetwork


class TemporalFusionTransformer(nn.Module):
    """
    Temporal Fusion Transformer (TFT) for interpretable multi-horizon forecasting.
    
    Key components:
    - Variable Selection Networks for input feature weighting
    - LSTM encoder-decoder for local temporal processing
    - Interpretable Multi-Head Attention for long-range dependencies
    - Gated Residual Networks for non-linear processing
    - Quantile outputs for probabilistic forecasting
    
    Reference: Lim et al. (2019) "Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting"
    """
    
    def __init__(
        self,
        # Architecture parameters
        hidden_size: int = 128,
        attention_head_count: int = 4,
        lstm_layers: int = 2,
        dropout: float = 0.1,
        
        # Input specifications
        input_size_price: int = 5,      # OHLCV + derived features
        input_size_news: int = 768,     # FinBERT embedding size
        static_categorical_size: int = 1,  # Asset type embedding
        static_real_size: int = 0,
        time_varying_known_categorical_size: int = 0,
        time_varying_known_real_size: int = 3,  # hour, day_of_week, is_news_day
        time_varying_unknown_categorical_size: int = 0,
        time_varying_unknown_real_size: int = 3,  # return, volatility, news_embed
        
        # Output specifications
        prediction_length: int = 1,
        quantiles: List[float] = [0.05, 0.5, 0.95],
        
        # Gating parameters
        gate_additive: bool = True,
        gate_multiplier: float = 2.0,
        
        # Variable selection parameters
        variable_selection_hidden: int = 64,
        variable_selection_dropout: float = 0.1,
        
        # Misc
        layer_norm_eps: float = 1e-5
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.attention_head_count = attention_head_count
        self.lstm_layers = lstm_layers
        self.dropout = dropout
        self.prediction_length = prediction_length
        self.quantiles = quantiles
        
        # Combined input sizes
        self.time_varying_unknown_total = (
            time_varying_unknown_real_size + 
            input_size_news  # News embeddings treated as unknown real
        )
        self.time_varying_known_total = time_varying_known_real_size
        self.static_total = static_categorical_size + static_real_size
        
        # ===== Variable Selection Networks =====
        # For time-varying unknown inputs (price features + news embeddings)
        self.vsn_unknown = VariableSelectionNetwork(
            input_size=self.time_varying_unknown_total,
            hidden_size=variable_selection_hidden,
            dropout=variable_selection_dropout,
            gate_additive=gate_additive,
            gate_multiplier=gate_multiplier
        )
        
        # For time-varying known inputs (time features)
        if self.time_varying_known_total > 0:
            self.vsn_known = VariableSelectionNetwork(
                input_size=self.time_varying_known_total,
                hidden_size=variable_selection_hidden,
                dropout=variable_selection_dropout,
                gate_additive=gate_additive,
                gate_multiplier=gate_multiplier
            )
        
        # For static inputs
        if self.static_total > 0:
            self.vsn_static = VariableSelectionNetwork(
                input_size=self.static_total,
                hidden_size=variable_selection_hidden,
                dropout=variable_selection_dropout,
                gate_additive=gate_additive,
                gate_multiplier=gate_multiplier
            )
            self.static_enrichment = GatedResidualNetwork(
                input_size=hidden_size,
                hidden_size=hidden_size,
                dropout=dropout,
                gate_additive=gate_additive
            )
        
        # ===== Input projection layers =====
        # Project selected unknown features to hidden size
        self.unknown_projection = nn.Linear(
            self.time_varying_unknown_total, hidden_size
        )
        
        if self.time_varying_known_total > 0:
            self.known_projection = nn.Linear(
                self.time_varying_known_total, hidden_size
            )
        
        # ===== LSTM Encoder-Decoder =====
        self.lstm_encoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0
        )
        
        self.lstm_decoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0
        )
        
        # ===== Interpretable Multi-Head Attention =====
        self.interpretable_attention = InterpretableMultiHeadAttention(
            hidden_size=hidden_size,
            head_count=attention_head_count,
            dropout=dropout
        )
        
        # ===== Post-attention processing =====
        self.post_attn_grn = GatedResidualNetwork(
            input_size=hidden_size,
            hidden_size=hidden_size,
            dropout=dropout,
            gate_additive=gate_additive
        )
        
        self.layer_norm_attn = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        
        # ===== Position-wise feed-forward =====
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout)
        )
        
        self.layer_norm_ffn = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        
        # ===== Output layer =====
        # Quantile regression outputs
        self.quantile_outputs = nn.ModuleList([
            nn.Linear(hidden_size, prediction_length) 
            for _ in quantiles
        ])
        
        # Optional: directional classification head
        self.directional_head = nn.Linear(hidden_size, 2)
        
        # Store attention weights for interpretability
        self._attention_weights = None
        self._variable_selection_weights = None
    
    def forward(
        self,
        # Time-varying unknown (changes during forecast horizon)
        unknown_inputs: torch.Tensor,  # [batch, time, features]
        unknown_mask: torch.Tensor,     # [batch, time]
        
        # Time-varying known (known during forecast horizon)
        known_inputs: Optional[torch.Tensor] = None,  # [batch, time, features]
        known_mask: Optional[torch.Tensor] = None,
        
        # Static inputs (constant across time)
        static_inputs: Optional[torch.Tensor] = None,  # [batch, features]
        
        # Decoder length (for forecasting)
        decoder_length: Optional[int] = None,
        
        # Return interpretability outputs
        return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through TFT.
        
        Args:
            unknown_inputs: Time-varying unknown features (price + news)
            unknown_mask: Mask for unknown inputs (1=valid, 0=padded)
            known_inputs: Time-varying known features (time indicators)
            known_mask: Mask for known inputs
            static_inputs: Static features (asset type, etc.)
            decoder_length: Length of decoder sequence for forecasting
            return_attention: Whether to return attention weights
            
        Returns:
            Dictionary with:
            - 'predictions': Quantile predictions [batch, n_quantiles, prediction_length]
            - 'directional_logits': Direction classification logits [batch, 2]
            - 'attention_weights': Attention weights if return_attention=True
            - 'variable_selection_weights': Feature selection weights if return_attention=True
        """
        batch_size, encoder_length, _ = unknown_inputs.shape
        device = unknown_inputs.device
        
        # ===== Variable Selection =====
        # Process unknown inputs
        selected_unknown, vsn_unknown_weights = self.vsn_unknown(
            unknown_inputs, unknown_mask
        )  # [B, T, H], [B, T, input_dim]
        
        # Process known inputs if provided
        if known_inputs is not None and self.time_varying_known_total > 0:
            selected_known, vsn_known_weights = self.vsn_known(
                known_inputs, known_mask
            )
        else:
            selected_known = None
            vsn_known_weights = None
        
        # Process static inputs if provided
        if static_inputs is not None and self.static_total > 0:
            selected_static, vsn_static_weights = self.vsn_static(
                static_inputs
            )  # [B, H]
            # Enrich encoder/decoder with static context
            static_context = self.static_enrichment(selected_static)
        else:
            static_context = None
            vsn_static_weights = None
        
        # ===== Input Projection =====
        # Project selected features to hidden dimension
        unknown_projected = self.unknown_projection(selected_unknown)
        
        if selected_known is not None:
            known_projected = self.known_projection(selected_known)
            # Combine known and unknown
            combined = unknown_projected + known_projected
        else:
            combined = unknown_projected
        
        # Add static context if available
        if static_context is not None:
            combined = combined + static_context.unsqueeze(1)
        
        # ===== LSTM Encoder =====
        encoder_outputs, (hidden, cell) = self.lstm_encoder(
            combined, None
        )  # [B, T_enc, H], (h: [L, B, H], c: [L, B, H])
        
        # ===== LSTM Decoder =====
        decoder_len = decoder_length if decoder_length is not None else 1
        decoder_input = encoder_outputs[:, -1:].repeat(1, decoder_len, 1)
        
        decoder_outputs, _ = self.lstm_decoder(
            decoder_input, (hidden, cell)
        )  # [B, T_dec, H]
        
        # ===== Interpretable Multi-Head Attention =====
        # Use encoder outputs as keys/values, decoder outputs as queries
        attn_output, attn_weights = self.interpretable_attention(
            query=decoder_outputs,
            key=encoder_outputs,
            value=encoder_outputs,
            mask=unknown_mask
        )  # [B, T_dec, H], [B, heads, T_dec, T_enc]
        
        # Store attention weights for interpretability
        if return_attention:
            self._attention_weights = attn_weights.detach()
            self._variable_selection_weights = {
                'unknown': vsn_unknown_weights.detach(),
                'known': vsn_known_weights.detach() if vsn_known_weights is not None else None,
                'static': vsn_static_weights.detach() if vsn_static_weights is not None else None
            }
        
        # ===== Post-Attention Processing =====
        # Gated residual connection
        attn_output = self.post_attn_grn(attn_output)
        attn_output = self.layer_norm_attn(
            decoder_outputs + attn_output  # Residual connection
        )
        
        # ===== Feed-Forward Network =====
        ffn_output = self.ffn(attn_output)
        output = self.layer_norm_ffn(attn_output + ffn_output)
        
        # ===== Output Heads =====
        # Quantile predictions
        predictions = torch.stack([
            head(output[:, -1, :])  # Use last decoder step
            for head in self.quantile_outputs
        ], dim=1)  # [B, n_quantiles, prediction_length]
        
        # Directional classification
        directional_logits = self.directional_head(output[:, -1, :])
        
        result = {
            'predictions': predictions,
            'directional_logits': directional_logits
        }
        
        if return_attention:
            result['attention_weights'] = attn_weights
            result['variable_selection_weights'] = self._variable_selection_weights
        
        return result
    
    def get_attention_weights(self) -> Optional[Dict[str, torch.Tensor]]:
        """Retrieve stored attention weights after forward pass."""
        if self._attention_weights is None:
            return None
        
        return {
            'temporal_attention': self._attention_weights,
            'variable_selection': self._variable_selection_weights
        }
    
    def reset_attention_cache(self):
        """Clear stored attention weights."""
        self._attention_weights = None
        self._variable_selection_weights = None


class GatedResidualNetwork(nn.Module):
    """
    Gated Residual Network (GRN) from TFT.
    
    Provides flexible non-linear processing with gating mechanism
    to skip unnecessary transformations.
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        dropout: float = 0.1,
        gate_additive: bool = True
    ):
        super().__init__()
        
        self.gate_additive = gate_additive
        
        # Linear projections
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        
        # Gating mechanism
        self.gate = nn.Linear(input_size, hidden_size)
        
        # Layer norm and dropout
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        
        # ELU activation
        self.activation = nn.ELU()
    
    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through GRN.
        
        Args:
            x: Input tensor
            context: Optional context for gating (if None, uses x)
        """
        if context is None:
            context = x
        
        # Main transformation
        a = self.linear1(x)
        a = self.activation(a)
        a = self.linear2(a)
        a = self.dropout(a)
        
        # Gate
        g = torch.sigmoid(self.gate(context))
        
        # Apply gate
        if self.gate_additive:
            output = self.layer_norm(x + g * a)
        else:
            output = self.layer_norm(g * a)
        
        return output
