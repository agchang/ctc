import torch
from functools import cache


@cache
def _gaa(s: int, t, last=0):
    """
    e.g. _gaa(s = 4, t = "dog")
             ε  + _gaa(s = 3, t="dog")
            "d" + _gaa(s = 3, t="dog")
            "d" + _gaa(s = 3, t="og")
                 ε  + _gaa(s = 2, t="og")
                "o" + _gaa(s = 2, t="og")
                "o" + _gaa(s = 2, t="g")
                     ε  + _gaa(s = 1, t="g")
                    "g" + _gaa(s = 1, t="g")
                    "g" + _gaa(s = 1, t="")

    e.g. _gaa(s=4, t="aεa")
            ε + _gaa(3, t="aεa")
            a + _gaa(3, t="aεa")
                ε + _gaa(2, t="a")
                a + _gaa(2, t="aεa")
                a + _gaa(2, t="εa")
            a + _gaa(3, t="εa")
    """
    alignments = []
    # base cases
    if len(t) == 0:
        return [torch.full((s,), 0)]
    if len(t) == s:
        # there is only one alignment, and that is the target seq itself
        return [t]
    elif len(t) > s:
        return []
    else:
        for a in _gaa(s - 1, t, last=t[0]):
            alignments.append(torch.concat((torch.tensor([t[0]]), a)))

        if len(t) >= 2 and t[1] == 0:
            for a in _gaa(s - 2, t[2:], last=0):
                alignments.append(torch.concat((torch.tensor([t[0], t[1]]), a)))
        else:
            for a in _gaa(s - 1, t[1:], last=t[0]):
                alignments.append(torch.concat((torch.tensor([t[0]]), a)))

        if len(t) >= 2 and torch.all(t[:2] == torch.tensor([last, 0])):
            for a in _gaa(s - 2, t[2:], last=0):
                alignments.append(torch.concat((torch.tensor([t[0], t[1]]), a)))
        elif last != t[0]:
            for a in _gaa(s - 1, t, last=0):
                alignments.append(torch.concat((torch.tensor([0]), a)))
    return [t for t in torch.unique(torch.vstack(alignments), dim=0)]


def naive_ctc_loss(
    log_probs,
    targets,
    input_lengths,
    target_lengths,
    blank=0,
    reduction="mean",
    zero_infinity=False,
):
    """
    A naive implementation of CTC loss that generates all possible alignments to the target string given
    an input length S.

    This algorithm is exponential, so we clearly need to do better as this will be called at every minibatch.

    log_probs - (T, N, C) where T is sequence length, N is batch size, and C is number of classes
    targets - (N, S) where S is the max target size in a batch (so smaller targets must be padded)
    input_lengths - (N) lengths of inputs, each must be <= T
    target_lengths - (N) lengths of targets, each must be <= S
    """
    losses = []
    for i, t_len in enumerate(target_lengths):
        target = targets[i][:t_len]
        i_len = input_lengths[i]
        log_probs_single = log_probs[:i_len, i, :]

        # insert ε between all repeats
        target_new = torch.tensor([], dtype=torch.int64)
        for i in range(len(target)):
            cur = target[i]
            target_new = torch.cat((target_new, torch.tensor([cur])), 0)
            if i == len(target) - 1:
                break
            nex = target[i + 1]
            if cur == nex:
                target_new = torch.cat((target_new, torch.tensor([0])), 0)
        target = target_new

        # generate all alignments of sequence len `i_len` to `target`
        alignments = _gaa(i_len, target)
        loss_single = torch.tensor(0.0)
        for a in alignments:
            a_log_prob = torch.sum(log_probs_single.gather(1, a.view(-1, 1)))
            loss_single += torch.exp(a_log_prob)

        losses.append(-torch.log(loss_single))

    return torch.tensor(losses)
