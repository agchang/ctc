import torch
from torchvision import datasets, transforms

from main import Net2, test


model = Net2()
model.load_state_dict(torch.load("mnist_cnn.pt"))

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
)
dataset2 = datasets.MNIST("data", train=False, transform=transform)

test_loader = torch.utils.data.DataLoader(dataset2)
device = "cpu"
test(model, device, test_loader)
