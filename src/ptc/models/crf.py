"""Linear-chain CRF (plan D7), written out rather than imported so it's auditable.

Masks must be left-aligned prefixes (True for real tokens, then padding). Emissions
are computed in float32 regardless of autocast.
"""
from __future__ import annotations

import torch
from torch import nn

NEG = -1e4


def bio_constraints(num_tags: int = 3) -> tuple[torch.Tensor, torch.Tensor]:
    """Forbid O->I and starting on I (tags: O=0, B=1, I=2)."""
    allowed = torch.ones(num_tags, num_tags, dtype=torch.bool)
    allowed[0, 2] = False
    start_allowed = torch.ones(num_tags, dtype=torch.bool)
    start_allowed[2] = False
    return allowed, start_allowed


class CRF(nn.Module):
    def __init__(self, num_tags: int, allowed: torch.Tensor | None = None,
                 start_allowed: torch.Tensor | None = None):
        super().__init__()
        self.num_tags = num_tags
        self.transitions = nn.Parameter(torch.empty(num_tags, num_tags).uniform_(-0.1, 0.1))  # [from, to]
        self.start_transitions = nn.Parameter(torch.empty(num_tags).uniform_(-0.1, 0.1))
        self.end_transitions = nn.Parameter(torch.empty(num_tags).uniform_(-0.1, 0.1))
        tmask = torch.zeros(num_tags, num_tags)
        smask = torch.zeros(num_tags)
        if allowed is not None:
            tmask[~allowed] = NEG
        if start_allowed is not None:
            smask[~start_allowed] = NEG
        self.register_buffer("_tmask", tmask, persistent=False)
        self.register_buffer("_smask", smask, persistent=False)

    def _trans(self) -> torch.Tensor:
        return self.transitions + self._tmask

    def _start(self) -> torch.Tensor:
        return self.start_transitions + self._smask

    # ------------------------------------------------------------------ training
    def neg_log_likelihood(self, emissions: torch.Tensor, tags: torch.Tensor,
                           mask: torch.Tensor) -> torch.Tensor:
        emissions = emissions.float()
        mask = mask.bool()
        return (self._log_partition(emissions, mask) - self._score(emissions, tags, mask)).mean()

    def _score(self, em: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        em_scores = em.gather(2, tags.unsqueeze(2)).squeeze(2)          # B,T
        score = self._start()[tags[:, 0]] + em_scores[:, 0]
        if em.size(1) > 1:
            trans = self._trans()[tags[:, :-1], tags[:, 1:]]            # B,T-1
            score = score + ((trans + em_scores[:, 1:]) * mask[:, 1:].float()).sum(1)
        last = tags.gather(1, (mask.long().sum(1) - 1).unsqueeze(1)).squeeze(1)
        return score + self.end_transitions[last]

    def _forward_alphas(self, em: torch.Tensor, mask: torch.Tensor) -> list[torch.Tensor]:
        trans = self._trans()
        alpha = self._start() + em[:, 0]
        alphas = [alpha]
        for t in range(1, em.size(1)):
            nxt = torch.logsumexp(alpha.unsqueeze(2) + trans.unsqueeze(0), dim=1) + em[:, t]
            alpha = torch.where(mask[:, t].unsqueeze(1), nxt, alpha)
            alphas.append(alpha)
        return alphas

    def _log_partition(self, em: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return torch.logsumexp(self._forward_alphas(em, mask)[-1] + self.end_transitions, dim=1)

    # ----------------------------------------------------------------- inference
    @torch.no_grad()
    def decode(self, emissions: torch.Tensor, mask: torch.Tensor) -> list[list[int]]:
        em, mask = emissions.float(), mask.bool()
        B, T, K = em.shape
        trans = self._trans()
        alpha = self._start() + em[:, 0]
        identity = torch.arange(K, device=em.device).expand(B, K)
        history = []
        for t in range(1, T):
            best, idx = (alpha.unsqueeze(2) + trans.unsqueeze(0)).max(dim=1)
            m = mask[:, t].unsqueeze(1)
            alpha = torch.where(m, best + em[:, t], alpha)
            history.append(torch.where(m, idx, identity))  # padding: carry the tag through
        cur = (alpha + self.end_transitions).argmax(dim=1)
        path = [cur]
        for bp in reversed(history):
            cur = bp.gather(1, cur.unsqueeze(1)).squeeze(1)
            path.append(cur)
        tags = torch.stack(path[::-1], dim=1)
        lengths = mask.long().sum(1).tolist()
        return [tags[b, :lengths[b]].tolist() for b in range(B)]

    @torch.no_grad()
    def marginals(self, emissions: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Posterior P(tag_t = k | x) via forward-backward. Returns B,T,K (padding rows arbitrary)."""
        em, mask = emissions.float(), mask.bool()
        B, T, K = em.shape
        trans = self._trans()
        alphas = self._forward_alphas(em, mask)
        log_z = torch.logsumexp(alphas[-1] + self.end_transitions, dim=1)
        end = self.end_transitions.unsqueeze(0).expand(B, K)
        beta = end
        betas = [beta]
        for t in range(T - 2, -1, -1):
            nxt = torch.logsumexp(trans.unsqueeze(0) + (em[:, t + 1] + beta).unsqueeze(1), dim=2)
            beta = torch.where(mask[:, t + 1].unsqueeze(1), nxt, end)
            betas.append(beta)
        betas.reverse()
        log_marg = torch.stack(alphas, 1) + torch.stack(betas, 1) - log_z.view(B, 1, 1)
        return log_marg.exp()
