import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision.models as models
from torch.optim.lr_scheduler import StepLR

from loss.dp_ctc_loss import dp_ctc_loss


class Net3(nn.Module):
    def __init__(self):
        super(Net3, self).__init__()
        self.vgg16 = models.vgg16()
        self.vgg16.features[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.vgg16.classifier = nn.Conv2d(512, 11, kernel_size=(1, 1))

    def forward(self, x):
        x = self.vgg16(x)
        print(x.shape)
        x = torch.mean(x, 2)  # (N, 11, 5)
        # NCW
        output = F.log_softmax(x, dim=1)
        return output

class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        self.resnet18 = models.resnet18()
        self.resnet18.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.resnet18.fc = nn.Conv2d(512, 11, kernel_size=(1, 1))

    def forward(self, x):
        # Copied from _forward_impl in order to remove the
        # torch.flatten() call.
        x = self.resnet18.conv1(x)
        x = self.resnet18.bn1(x)
        x = self.resnet18.relu(x)
        x = self.resnet18.maxpool(x)

        x = self.resnet18.layer1(x)
        x = self.resnet18.layer2(x)
        x = self.resnet18.layer3(x)
        x = self.resnet18.layer4(x)

        x = self.resnet18.avgpool(x)
        x = self.resnet18.fc(x)
        x = torch.mean(x, 2)  # (N, 11, 5)
        # NCW
        output = F.log_softmax(x, dim=1)
        return output


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1)
        self.conv4 = nn.Conv2d(128, 256, 3, 1)
        self.conv5 = nn.Conv2d(256, 11, 1) 

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 3)
        x = self.conv5(x)
        # NCHW
        x = torch.mean(x, 2)  # (N, 11, 5)
        # NCW
        output = F.log_softmax(x, dim=1)
        return output


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        resize = transforms.Resize((224, 224))
        data = resize(data)
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        # torch.Size([64, 1, 28, 28])
        # NCHW
        output = model(data)
        seq_len = output.shape[2]
        #print(f"seq_len: {seq_len}")
        output = output.reshape((seq_len, data.shape[0], 11))
        # todo: variable width
        input_lengths = torch.full(size=(data.shape[0],), fill_value=seq_len)
        target_lengths = torch.full(size=(data.shape[0],), fill_value=1)
        #loss = F.ctc_loss(output, target, input_lengths, target_lengths, blank=10)
        loss = dp_ctc_loss(output, target, input_lengths, target_lengths, blank=10, reduction='mean')
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def decode(logprobs):
    outs = []
    for i in range(logprobs.shape[1]):
        labels = torch.argmax(logprobs[:,i,:], dim=1)
        labels = torch.unique_consecutive(labels)
        labels = labels[labels != 10]
        if len(labels) == 0:
            outs.append(-1)
        else:
            decoded = int("".join(str(x) for x in [int(x) for x in labels]))
            outs.append(decoded)
    return torch.tensor(outs)


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            resize = transforms.Resize((224, 224))
            data = resize(data)

            output = model(data)  # (N, 11, 5)
            seq_len = output.shape[2]
            output = output.reshape((seq_len, data.shape[0], 11))
            input_lengths = torch.full(size=(data.shape[0],), fill_value=seq_len)
            target_lengths = torch.full(size=(data.shape[0],), fill_value=1)
            #test_loss += F.ctc_loss(output, target, input_lengths, target_lengths, blank=10, reduction='sum').item()  # sum up batch loss
            test_loss += dp_ctc_loss(output, target, input_lengths, target_lengths, blank=10, reduction='sum').item()  # sum up batch loss
            pred = decode(output)
            #print(f"target: {target}")
            #print(f"pred: {pred}")
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset1 = datasets.MNIST('data', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.MNIST('data', train=False,
                       transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = Net2().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()

        if args.save_model:
            torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    main()
