import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler

import torchvision.datasets as dset
import torchvision.transforms as T

from resnet_classifier import ResNet


def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()  
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device, dtype=torch.float32)
            y = y.to(device=device, dtype=torch.long)
            scores = model(x)
            _, preds = scores.max(1)
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)
        acc = float(num_correct) / num_samples
        print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))
    return acc


NUM_TRAIN = 49000
batch_size= 64
if __name__ == '__main__':
    train_transform = transform = T.Compose([
        T.ToTensor(),
        T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])

    cifar10_train = dset.CIFAR10('./datasets/cifar10', train=True, download=True,
        transform=train_transform)
    loader_train = DataLoader(cifar10_train, batch_size=batch_size, num_workers=2,
        sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN)))

    cifar10_val = dset.CIFAR10('./datasets/cifar10', train=True, download=True,
        transform=transform)
    loader_val = DataLoader(cifar10_val, batch_size=batch_size, num_workers=2, 
        sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN, 50000)))

    cifar10_test = dset.CIFAR10('./datasets/cifar10', train=False, download=True, 
        transform=transform)
    loader_test = DataLoader(cifar10_test, batch_size=batch_size, num_workers=2)


    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    model = nn.Sequential(
        nn.Conv2d(3, 64, 7, stride = 2, padding = 3),
        nn.ReLU(),
        nn.MaxPool2d(3, stride=2, padding = 1),
        ResNet(64,64,3,padding=1),
        ResNet(64,128,3,padding=1),
        ResNet(128,256,3,padding=1),
        ResNet(256,512,3,padding=1,downsample=True),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(512, 10)
    )

    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)

    model = model.to(device=device)  # move the model parameters to CPU/GPU
    loss_func = nn.CrossEntropyLoss()
    epochs = 10

    for e in range(epochs):
        for t, (x, y) in enumerate(loader_train):
            model.train()  # put model to training mode
            x = x.to(device=device, dtype=torch.float32)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=torch.long)

            scores = model(x)

            optimizer.zero_grad()
            loss = loss_func(scores, y)
            # Backward pass
            loss.backward()
            # Optimization step
            optimizer.step()
        check_accuracy(loader_val, model)
        print(f'Epoch {e+1}, Loss: {loss.item():.4f}')
    torch.save(model.state_dict(), 'classifier.pth')

