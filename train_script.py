import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torch.utils.data.dataloader import DataLoader
import numpy as np
from torchvision.models import resnet18
from torchvision.datasets import CIFAR10

net = resnet18(pretrained=False)
net.load_state_dict(torch.load('model.pt'))

cif = CIFAR10('.', download=True, transform=transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()]))
dl = DataLoader(cif, batch_size=128)

lr = 0.001
xent = torch.nn.CrossEntropyLoss()
optim = torch.optim.Adam(net.parameters(), lr=lr)
n_epochs = 5
for epoch in range(n_epochs):
    for i, (x, y) in enumerate(dl):
        optim.zero_grad()
        out = net(x)
        loss = xent(out, y)
        loss.backward()
        optim.step()
        if i % 10 == 0:
            print('Epoch: {}/{}, Batch: {}/{}, Loss: {}'.format(epoch + 1, n_epochs, i + 1, len(dl), loss))

            
torch.save(net.state_dict(), 'model.pt')