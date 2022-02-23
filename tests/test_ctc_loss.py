import torch
import torch.nn.functional as F
from loss import naive_ctc_loss


def test_ctc_loss_trivial():
    T = 1  # input seq length
    C = 2  # [ε, a]
    S = 1

    input = (
        torch.log(
            torch.tensor(
                [
                    [0.2, 0.8],
                ]
            )
        )
        .reshape(T, 1, C)
        .detach()
        .requires_grad_()
    )

    target = torch.tensor([1]).reshape(1, S)
    input_lengths = torch.tensor([T])
    target_lengths = torch.tensor([S])

    # All outputs that align to "a" are
    # a (log(0.8)) = -0.2231
    ref_loss = F.ctc_loss(input, target, input_lengths, target_lengths, reduction="none")
    loss = naive_ctc_loss.naive_ctc_loss(input, target, input_lengths, target_lengths, reduction="none")

    assert torch.allclose(loss, ref_loss)
    # loss.backward()


def test_ctc_loss_basic():
    T = 2  # input seq length
    C = 2  # [ε, a]
    S = 1

    input = (
        torch.log(
            torch.tensor(
                [
                    [0.2, 0.8],
                    [0.4, 0.6],
                ]
            )
        )
        .reshape(T, 1, C)
        .detach()
        .requires_grad_()
    )

    target = torch.tensor([1]).reshape(1, S)
    input_lengths = torch.tensor([T])
    target_lengths = torch.tensor([S])

    # All outputs that align to "a" are
    # aa - log(0.8) + log(0.6)
    # aε - log(0.8) + log(0.4)
    # εa - log(0.2) + log(0.6)
    # We need to go back to non-log space to add probabilities, so:
    # np.exp(log(0.8) + log(0.6)) + np.exp(log(0.8) + log(0.4)) + np.exp(log(0.2) + log(0.6)) = 0.92
    # Back to logspace:
    # np.log(0.92) = -0.0833
    ref_loss = F.ctc_loss(input, target, input_lengths, target_lengths, reduction="none")
    loss = naive_ctc_loss.naive_ctc_loss(input, target, input_lengths, target_lengths, reduction="none")

    assert torch.allclose(loss, ref_loss)

    # loss.backward()


def test_ctc_loss_repeats():
    T = 3  # input seq length
    C = 2  # [ε, a]
    S = 2

    input = (
        torch.log(
            torch.tensor(
                [
                    [0.2, 0.8],
                    [0.4, 0.6],
                    [0.3, 0.7],
                ]
            )
        )
        .reshape(T, 1, C)
        .detach()
        .requires_grad_()
    )

    target = torch.tensor([1, 1]).reshape(1, S)
    input_lengths = torch.tensor([T])
    target_lengths = torch.tensor([S])

    # All outputs that align to "aa" are
    # aεa - log(0.8) + log(0.4) + log(0.7) = -1.4961
    ref_loss = F.ctc_loss(input, target, input_lengths, target_lengths, reduction="none")
    loss = naive_ctc_loss.naive_ctc_loss(input, target, input_lengths, target_lengths, reduction="none")

    assert torch.allclose(loss, ref_loss)

    # loss.backward()


def test_ctc_loss():
    T = 4  # input seq length
    C = 4  # [ε, c, a, t]
    S = 3

    input = (
        torch.log(
            torch.tensor(
                [
                    [0.8, 0.1, 0.1, 0.0],
                    [0.1, 0.7, 0.1, 0.1],
                    [0.0, 0.3, 0.5, 0.2],
                    [0.1, 0.1, 0.2, 0.6],
                ]
            )
        )
        .reshape(T, 1, C)
        .detach()
        .requires_grad_()
    )

    target = torch.tensor([1, 2, 3]).reshape(1, S)  # c, a, t
    input_lengths = torch.tensor([T])
    target_lengths = torch.tensor([S])

    # All outputs that align to "cat" are
    # ccat (log(0.1) + log(0.7) + log(0.5) + log(0.6)) = -3.863
    # caat (log(0.1) + log(0.1) + log(0.5) + log(0.6)) = -5.809
    # catt (log(0.1) + log(0.1) + log(0.2) + log(0.6)) = -6.725
    # εcat (log(0.8) + log(0.7) + log(0.5) + log(0.6)) = -1.783
    # cεat (log(0.1) + log(0.1) + log(0.5) + log(0.6)) = -5.809
    # caεt (log(0.1) + log(0.1) + log(0.0) + log(0.6)) = -inf
    # catε (log(0.1) + log(0.1) + log(0.2) + log(0.1)) = -8.517193191416236
    # np.exp(-3.863) + np.exp(-5.809) + np.exp(-6.725) +
    #   np.exp(-1.783) + np.exp(-5.809) + 0.0 + np.exp(-8.517193191416236) = 0.1965
    # np.log(0.1965) = -1.626
    ref_loss = F.ctc_loss(input, target, input_lengths, target_lengths, reduction="none")
    loss = naive_ctc_loss.naive_ctc_loss(input, target, input_lengths, target_lengths, reduction="none")

    assert torch.allclose(loss, ref_loss)
    # loss.backward()


def test_ctc_loss_random_batch():
    T = 5
    C = 5
    S = 5
    S_min = 2
    N = 2

    input = torch.randn(T, N, C).log_softmax(2).detach().requires_grad_()
    target = torch.randint(low=1, high=C, size=(N, S), dtype=torch.long)
    input_lengths = torch.full(size=(N,), fill_value=T, dtype=torch.long)
    target_lengths = torch.randint(low=S_min, high=S, size=(N,), dtype=torch.long)

    ref_loss = F.ctc_loss(input, target, input_lengths, target_lengths, reduction="none")
    loss = naive_ctc_loss.naive_ctc_loss(input, target, input_lengths, target_lengths, reduction="none")

    assert torch.allclose(loss, ref_loss)
