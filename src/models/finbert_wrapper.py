"""
FinBERT wrapper for financial news encoding.
"""

import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel, AutoTokenizer, AutoModel
from typing import Dict, List, Optional, Tuple, Union
from loguru import logger


class FinBERTWrapper(nn.Module):
    """
    Wrapper around FinBERT for encoding financial news headlines.
    
    Provides:
    - Contextual embeddings (CLS token or pooled)
    - Sentiment scores (optional head)
    - Token-level attention weights for interpretability
    """
    
    def __init__(
        self,
        model_name: str = "ProsusAI/finbert",
        max_length: int = 128,
        freeze_embeddings: bool = False,
        freeze_layers: int = 0,
        pooling: str = "cls",
        add_sentiment_head: bool = True,
        cache_dir: Optional[str] = None
    ):
        super().__init__()
        
        self.model_name = model_name
        self.max_length = max_length
        self.pooling = pooling
        self.add_sentiment_head = add_sentiment_head
        
        # Load tokenizer and model
        logger.info(f"Loading FinBERT: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, cache_dir=cache_dir
        )
        self.bert = AutoModel.from_pretrained(
            model_name, cache_dir=cache_dir, output_attentions=True
        )
        
        # Freeze layers if specified
        if freeze_embeddings:
            for param in self.bert.embeddings.parameters():
                param.requires_grad = False
        
        if freeze_layers > 0:
            for i in range(freeze_layers):
                for param in self.bert.encoder.layer[i].parameters():
                    param.requires_grad = False
            logger.info(f"Froze first {freeze_layers} BERT layers")
        
        # Hidden size
        self.hidden_size = self.bert.config.hidden_size
        
        # Optional sentiment classification head
        if add_sentiment_head:
            self.sentiment_classifier = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(self.hidden_size // 2, 3)  # negative, neutral, positive
            )
        
        # Register buffer for special token ids
        self.register_buffer(
            'pad_token_id', 
            torch.tensor(self.tokenizer.pad_token_id)
        )
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        return_dict: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through FinBERT.
        
        Args:
            input_ids: Token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            token_type_ids: Segment IDs (optional)
            return_dict: Return dict vs tuple
            
        Returns:
            Dictionary with:
            - 'embedding': Pooled embedding [batch, hidden_size]
            - 'token_embeddings': Per-token embeddings [batch, seq_len, hidden_size]
            - 'attention_weights': Attention weights from last layer [batch, heads, seq_len, seq_len]
            - 'sentiment': Sentiment logits [batch, 3] (if enabled)
        """
        # BERT forward pass
        bert_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_attentions=True
        )
        
        # Extract outputs
        sequence_output = bert_output.last_hidden_state  # [B, L, H]
        attention_weights = bert_output.attentions[-1]    # [B, heads, L, L]
        
        # Pooling
        if self.pooling == "cls":
            pooled = sequence_output[:, 0, :]  # [B, H]
        elif self.pooling == "mean":
            # Mean pooling over non-padded tokens
            mask_expanded = attention_mask.unsqueeze(-1).expand(
                sequence_output.size()
            ).float()
            pooled = torch.sum(
                sequence_output * mask_expanded, dim=1
            ) / torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        elif self.pooling == "max":
            mask_expanded = attention_mask.unsqueeze(-1).expand(
                sequence_output.size()
            ).float()
            sequence_output_masked = sequence_output.masked_fill(
                mask_expanded == 0, -1e9
            )
            pooled = torch.max(sequence_output_masked, dim=1)[0]
        else:
            raise ValueError(f"Unknown pooling method: {self.pooling}")
        
        result = {
            'embedding': pooled,
            'token_embeddings': sequence_output,
            'attention_weights': attention_weights
        }
        
        # Sentiment head
        if self.add_sentiment_head:
            result['sentiment'] = self.sentiment_classifier(pooled)
        
        if not return_dict:
            return tuple(result.values())
        
        return result
    
    def encode_texts(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        device: Optional[torch.device] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Encode text(s) to embeddings.
        
        Args:
            texts: Single string or list of strings
            batch_size: Batch size for encoding
            device: Target device
            
        Returns:
            Dictionary with embeddings and metadata
        """
        if isinstance(texts, str):
            texts = [texts]
        
        if device is None:
            device = next(self.parameters()).device
        
        all_embeddings = []
        all_sentiment = []
        all_tokens = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize
            encoded = self.tokenizer(
                batch_texts,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            # Move to device
            encoded = {k: v.to(device) for k, v in encoded.items()}
            
            # Forward pass
            with torch.no_grad():
                output = self(**encoded)
            
            all_embeddings.append(output['embedding'].cpu())
            if self.add_sentiment_head:
                all_sentiment.append(output['sentiment'].cpu())
            
            # Store tokens for interpretability
            batch_tokens = self.tokenizer.batch_decode(
                encoded['input_ids'], skip_special_tokens=True
            )
            all_tokens.extend(batch_tokens)
        
        result = {
            'embeddings': torch.cat(all_embeddings, dim=0),
            'tokens': all_tokens
        }
        
        if all_sentiment:
            result['sentiment'] = torch.cat(all_sentiment, dim=0)
        
        return result
    
    def get_token_attention(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Extract token-level attention weights for interpretability.
        
        Returns:
            Attention weights averaged over heads [batch, seq_len]
        """
        with torch.no_grad():
            output = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=True
            )
        
        # Get last layer attention [B, heads, L, L]
        last_layer_attn = output.attentions[-1]
        
        # Average over heads and over query positions (focus on CLS attention to tokens)
        # Shape: [B, L] - attention from CLS token to all tokens
        cls_attention = last_layer_attn[:, :, 0, :]  # [B, heads, L]
        token_attention = cls_attention.mean(dim=1)   # [B, L]
        
        # Mask out padding
        token_attention = token_attention * attention_mask
        
        return token_attention
