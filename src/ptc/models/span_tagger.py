"""Stage 1: encoder -> linear emissions -> CRF over O/B/I (plan D3, D7)."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .crf import CRF, bio_constraints


class SpanTagger(nn.Module):
    kind = "si"

    def __init__(self, encoder: nn.Module, use_crf: bool = True, dropout: float = 0.1, num_tags: int = 3):
        super().__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(encoder.config.hidden_size, num_tags)
        self.crf = CRF(num_tags, *bio_constraints(num_tags)) if use_crf else None

    def emissions(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        return self.proj(self.dropout(hidden)).float()

    def forward(self, input_ids, attention_mask, labels=None) -> dict:
        em = self.emissions(input_ids, attention_mask)
        out = {"emissions": em}
        if labels is not None:
            mask = attention_mask.bool()
            if self.crf is not None:
                with torch.autocast(device_type="cuda", enabled=False):
                    out["loss"] = self.crf.neg_log_likelihood(em.float(), labels, mask)
            else:
                out["loss"] = F.cross_entropy(em[mask], labels[mask])
        return out

    @torch.no_grad()
    def predict(self, input_ids, attention_mask) -> tuple[list[list[int]], torch.Tensor]:
        """Best tag path per sequence + per-token P(inside a span) = P(B) + P(I)."""
        em = self.emissions(input_ids, attention_mask)
        mask = attention_mask.bool()
        if self.crf is not None:
            paths = self.crf.decode(em, mask)
            probs = self.crf.marginals(em, mask)
        else:
            probs = em.softmax(-1)
            lengths = mask.long().sum(1).tolist()
            arg = em.argmax(-1)
            paths = [arg[b, :lengths[b]].tolist() for b in range(em.size(0))]
        return paths, probs[..., 1] + probs[..., 2]
