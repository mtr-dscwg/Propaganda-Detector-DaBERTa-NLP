import itertools

import torch

from ptc.models.crf import CRF, bio_constraints


def brute(crf, em, length):
    trans, start, end = crf._trans(), crf._start(), crf.end_transitions
    scores = {}
    for path in itertools.product(range(em.size(-1)), repeat=length):
        s = start[path[0]] + em[0, path[0]]
        for t in range(1, length):
            s = s + trans[path[t - 1], path[t]] + em[t, path[t]]
        scores[path] = s + end[path[-1]]
    return scores


def setup(seed=0):
    torch.manual_seed(seed)
    crf = CRF(3, *bio_constraints())
    with torch.no_grad():
        crf.transitions.normal_()
        crf.start_transitions.normal_()
        crf.end_transitions.normal_()
    em = torch.randn(2, 5, 3)
    mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]], dtype=torch.bool)
    return crf, em, mask


def test_partition_and_nll_match_brute_force():
    crf, em, mask = setup()
    tags = torch.tensor([[1, 2, 0, 1, 2], [0, 1, 2, 0, 0]])
    for b in range(2):
        L = int(mask[b].sum())
        scores = brute(crf, em[b], L)
        log_z = torch.logsumexp(torch.stack(list(scores.values())), 0)
        got_z = crf._log_partition(em[b:b + 1], mask[b:b + 1])
        assert torch.allclose(got_z, log_z.view(1), atol=1e-4)
        nll = crf.neg_log_likelihood(em[b:b + 1], tags[b:b + 1], mask[b:b + 1])
        assert torch.allclose(nll, log_z - scores[tuple(tags[b, :L].tolist())], atol=1e-4)


def test_viterbi_matches_brute_force():
    crf, em, mask = setup(1)
    paths = crf.decode(em, mask)
    for b in range(2):
        L = int(mask[b].sum())
        scores = brute(crf, em[b], L)
        assert tuple(paths[b]) == max(scores, key=lambda p: scores[p].item())
        assert 2 not in (paths[b][0],)  # never starts on I


def test_marginals_match_brute_force():
    crf, em, mask = setup(2)
    marg = crf.marginals(em, mask)
    for b in range(2):
        L = int(mask[b].sum())
        scores = brute(crf, em[b], L)
        probs = torch.softmax(torch.stack(list(scores.values())), 0)
        expected = torch.zeros(L, 3)
        for p, pr in zip(scores, probs):
            for t, k in enumerate(p):
                expected[t, k] += pr
        assert torch.allclose(marg[b, :L], expected, atol=1e-4)
        assert torch.allclose(marg[b, :L].sum(-1), torch.ones(L), atol=1e-4)
