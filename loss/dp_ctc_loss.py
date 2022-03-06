import torch
import numpy as np
from functools import cache


def logsumexp(it):
    c = max(it)
    if c == float("-inf"):
        return torch.tensor(float("-inf"))
    return c + torch.log(sum(torch.exp(i - c) for i in it))


def dp_ctc_loss(log_probs, targets, input_lengths, target_lengths, blank=0, reduction="mean", zero_infinity=False):
    """
    log_probs - (T, N, C) where T is sequence length, N is batch size, and C is number of classes
    targets - (N, S) where S is the max target size in a batch (so smaller targets must be padded)
    input_lengths - (N) lengths of inputs, each must be <= T
    target_lengths - (N) lengths of targets, each must be <= S
    """
    @cache
    def path_loss(i, t, s):
        # not enough characters left
        s_left = s // 2
        if t < s_left or (t == 0 and s_left > 0):
            return torch.tensor(float("-inf"))
        cur_char = eps_target[s]
        cur_prob = log_probs[t][i][cur_char]
        
        # base case
        if t == 0:
            return cur_prob
        paths = [cur_prob + path_loss(i, t - 1, s)]
        if s > 0:
            paths.append(cur_prob + path_loss(i, t - 1, s - 1))
        if cur_char == blank:
            return logsumexp(paths)
        else:
            if s > 1 and (cur_char != eps_target[s-2]):
                paths.append(cur_prob + path_loss(i, t - 1, s - 2))
            return logsumexp(paths)

    N = len(target_lengths)
    losses = []
    for i in range(N):
        T = input_lengths[i]
        S = target_lengths[i]
        if targets.ndim == 1:
            start = torch.sum(target_lengths[:i])
            target = targets[start:start+S]
        else:
            target = targets[i][:S]

        # Add eps after every char and before entire target, e.g.
        # ["c","a","t"] -> [0, "c", 0, "a", 0, "t", 0]
        eps_target = [blank]
        for x in target:
            eps_target.append(x)
            eps_target.append(blank)

        S = len(eps_target) - 1
        T = int(T) - 1
        #print(path_loss(T, S))
        #print(path_loss(T, S-1))
        loss = logsumexp([path_loss(i, T, S), path_loss(i, T, S - 1)])
        losses.append(-loss)
        #print(loss)
    losses = torch.cat([loss.unsqueeze(0) for loss in losses]).squeeze(0)
    if reduction == "none":
        return losses
    elif reduction == "mean":
        return torch.mean(losses)
    elif reduction == "sum":
        return torch.sum(losses)
