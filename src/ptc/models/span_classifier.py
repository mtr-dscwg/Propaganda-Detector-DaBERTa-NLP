"""Stage 2: encoder over `ctx [BOP] span [EOP] ctx` -> technique (plan D3, D7).

Representation = [CLS ; h(BOP) ; h(EOP)], a common entity-marker pooling.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class SpanClassifier(nn.Module):
    kind = "tc"

    def __init__(self, encoder: nn.Module, num_labels: int, dropout: float = 0.1,
                 class_weights: torch.Tensor | None = None):
        super().__init__()
        self.encoder = encoder
        h = encoder.config.hidden_size
        self.head = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(3 * h, h), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(h, num_labels),
        )
        self.register_buffer("class_weights", class_weights, persistent=False)

    def forward(self, input_ids, attention_mask, bop_pos, eop_pos, labels=None) -> dict:
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        rows = torch.arange(hidden.size(0), device=hidden.device)
        rep = torch.cat([hidden[:, 0], hidden[rows, bop_pos], hidden[rows, eop_pos]], dim=-1)
        logits = self.head(rep).float()
        out = {"logits": logits}
        if labels is not None:
            out["loss"] = F.cross_entropy(logits, labels, weight=self.class_weights)
        return out
