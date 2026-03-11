# coding=utf-8
"""
SASREC: Self-Attentive Sequential Recommendation
Based on Kang & McAuley, ICDM 2018

Implementation for strategy sequence prediction in ESConv.
Following ESCMamba's implementation style.

Token indexing:
- 0: [PAD]
- 1: [no_history]
- 2~9: Strategies

Output: 10 classes (same as input vocab, following ESCMamba)
Labels: 2~9 (strategies), ignore_index=0 (PAD)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SASREC(nn.Module):
    """
    SASREC: Self-Attentive Sequential Recommendation

    Following ESCMamba's implementation using PyTorch TransformerEncoder.

    For strategy sequence prediction:
    - Input: [no_history] + strategy sequence (IDs 0~9)
    - Output: predicted next strategy distribution at each position (10 classes)
    - Labels: strategy IDs (2~9), with ignore_index=0 for padding
    """

    def __init__(
        self,
        num_items: int = 10,  # Input vocab size: PAD + no_history + 8 strategies
        hidden_size: int = 256,
        num_blocks: int = 4,
        num_heads: int = 4,
        max_len: int = 50,
        dropout_rate: float = 0.2,
        output_num_items: int = None,  # Output classes (default: same as num_items)
    ):
        super().__init__()

        self.num_items = num_items
        self.hidden_size = hidden_size
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.max_len = max_len
        # Following ESCMamba: output_num_items = num_items (10)
        self.output_num_items = output_num_items if output_num_items is not None else num_items

        # Embedding layers (following ESCMamba)
        self.item_emb = nn.Embedding(num_items, hidden_size, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, hidden_size)
        self.emb_dropout = nn.Dropout(dropout_rate)

        # Transformer encoder (following ESCMamba)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout_rate,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_blocks)

        # Final prediction layer (following ESCMamba)
        self.final_layer = nn.Linear(hidden_size, self.output_num_items)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights following ESCMamba."""
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                module.weight.data.normal_(mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            input_ids: [batch_size, seq_len] - token IDs (0~9)
            attention_mask: [batch_size, seq_len] - 1 for valid, 0 for padding

        Returns:
            logits: [batch_size, seq_len, output_num_items]
        """
        seq_len = input_ids.size(1)
        device = input_ids.device

        # Position embedding
        positions = torch.arange(seq_len, dtype=torch.long, device=device)
        positions = positions.unsqueeze(0).expand_as(input_ids)

        # Embedding
        emb = self.item_emb(input_ids)  # [batch, seq, hidden]
        pos_emb = self.pos_emb(positions)  # [batch, seq, hidden]

        # Sum embeddings (following SASRec paper and ESCMamba)
        x = emb + pos_emb
        x = self.emb_dropout(x)

        # Create causal mask
        # [seq_len, seq_len] - float mask where -inf means "cannot see"
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device) * float('-inf'),
            diagonal=1
        )

        # Create padding mask for TransformerEncoder
        # PyTorch TransformerEncoder takes src_key_padding_mask as (True for padding)
        src_key_padding_mask = None
        if attention_mask is not None:
            src_key_padding_mask = (attention_mask == 0)

        # Forward pass through transformer
        output = self.encoder(
            x,
            mask=causal_mask,
            src_key_padding_mask=src_key_padding_mask
        )

        # Prediction
        logits = self.final_layer(output)  # [batch, seq, output_num_items]

        return logits

    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        actual_lens: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Predict next strategy given input sequence.

        Following ESCMamba: use the actual last position (not padded position).

        Args:
            input_ids: (batch_size, seq_len) strategy sequence
            attention_mask: (batch_size, seq_len) optional
            actual_lens: (batch_size,) actual sequence lengths (before padding)
                         If provided, uses logits[:, actual_len-1, :] for each sample
                         If None, uses logits[:, -1, :] (assumes no padding)

        Returns:
            probs: (batch_size, output_num_items) probability distribution
        """
        logits = self.forward(input_ids, attention_mask)  # [batch, seq, output_num_items]

        if actual_lens is not None:
            # Use actual last position for each sample (ESCMamba style)
            batch_size = logits.size(0)
            # Gather logits at actual_len - 1 for each sample
            last_indices = (actual_lens - 1).unsqueeze(-1).unsqueeze(-1)  # [batch, 1, 1]
            last_indices = last_indices.expand(-1, -1, logits.size(-1))  # [batch, 1, output_num_items]
            last_logits = logits.gather(1, last_indices).squeeze(1)  # [batch, output_num_items]
        else:
            # Assume no padding, use last position
            last_logits = logits[:, -1, :]  # [batch, output_num_items]

        probs = F.softmax(last_logits, dim=-1)

        return probs

    def compute_loss(
        self,
        input_ids: torch.Tensor,
        target_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
    ) -> dict:
        """
        Compute cross-entropy loss for next-item prediction.

        Following ESCMamba: ignore_index=0 (PAD token)

        Args:
            input_ids: (batch_size, seq_len) input strategy sequence
            target_ids: (batch_size, seq_len) target strategy IDs (2~9), padded with 0
            attention_mask: (batch_size, seq_len) optional

        Returns:
            dict with loss, accuracy, and logits
        """
        logits = self.forward(input_ids, attention_mask)  # [batch, seq, output_num_items]

        # Reshape for loss computation
        batch_size, seq_len, num_classes = logits.size()
        logits_flat = logits.view(-1, num_classes)  # [batch * seq, num_classes]
        targets_flat = target_ids.view(-1)  # [batch * seq]

        # Compute loss (ignore_index=0 for PAD, following ESCMamba)
        loss = F.cross_entropy(logits_flat, targets_flat, ignore_index=0)

        # Compute accuracy (only on valid positions, i.e., targets != 0)
        with torch.no_grad():
            valid_mask = targets_flat != 0
            if valid_mask.sum() > 0:
                predictions = logits_flat[valid_mask].argmax(dim=-1)
                targets_valid = targets_flat[valid_mask]
                accuracy = (predictions == targets_valid).float().mean()
            else:
                accuracy = torch.tensor(0.0, device=logits.device)

        return {
            'loss': loss,
            'accuracy': accuracy,
            'logits': logits,
        }
