import torch
import pdb
from torchviz import make_dot

from main import Net2
from loss.dp_ctc_loss import dp_ctc_loss

#net = Net2()

T = 20
C = 20
N = 16

input = torch.randn(T, N, C).log_softmax(2).detach().requires_grad_()
input_lengths = torch.full(size=(N,), fill_value=T, dtype=torch.long)
target_lengths = torch.randint(low=1, high=T, size=(N,), dtype=torch.long)
target = torch.randint(low=1, high=C, size=(sum(target_lengths),), dtype=torch.long)
loss = dp_ctc_loss(input, target, input_lengths, target_lengths)

#make_dot(loss).render("viz", format="png")
