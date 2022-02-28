import torch
import numpy as np
from functools import cache


def logsumexp(it):
    return torch.log(sum(torch.exp(i) for i in it))


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
        if cur_char == 0:
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
        target = targets[i][:S]

        # Add eps after every char and before entire target, e.g.
        # ["c","a","t"] -> [0, "c", 0, "a", 0, "t", 0]
        eps_target = [0]
        for x in target:
            eps_target.append(x)
            eps_target.append(0)

        S = len(eps_target) - 1
        T = int(T) - 1
        #print(path_loss(T, S))
        #print(path_loss(T, S-1))
        loss = logsumexp([path_loss(i, T, S), path_loss(i, T, S - 1)])
        losses.append(-loss)
        #print(loss)
    return torch.tensor(losses)
