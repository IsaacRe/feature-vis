import torch
import numpy as np
from PIL import Image
from argparse import ArgumentParser
import numpy as np
from torchvision.models import alexnet
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# means and stds for mnist and cifar
mean_MNIST = 0.1307
std_MNIST = 0.3081
mean_CIFAR = np.array([0.5071, 0.4867, 0.4408])
std_CIFAR = np.array([0.2675, 0.2565, 0.2761])


def input2image(img):
    if len(img.shape) > 2:
        img -= img.min(axis=(1, 2))[:, None, None]
        img *= 255. / img.max(axis=(1, 2))[:, None, None]
        img = np.transpose(img, (1, 2, 0))
        return Image.fromarray(img.astype(int), 'RGB')
    else:
        img -= img.min()
        img *= 255. / img.max()
        return Image.fromarray(img.astype(int), 'L')


class InputMap:
    """
    Class to handle visualization of features in a trained network by optimizing input to maximize their activation
    """

    def __init__(self, network, layer_idx, feat_idx, lr=10, num_updates=500, dataset='MNIST'):
        self.net = network
        self.layer_idx = layer_idx
        self.feat_idx = feat_idx
        self.lr = lr
        self.num_updates = num_updates
        self.dataset = dataset

    def _random_input_MNIST(self):
        """
        Returns random network input to be optimized
        :return: random input
        """
        img = np.random.uniform(0, 255, (1, 28, 28))
        img /= 255.
        img -= mean_MNIST
        img /= std_MNIST
        return torch.FloatTensor(img).requires_grad_(True)

    def _input2image_MNIST(self, img):
        # dereference batch dim
        img = img[0].data.clone().cpu().numpy()
        img *= std_MNIST
        img += mean_MNIST
        img *= 255.
        img[np.where(img > 255)] = 255
        return img.transpose(1, 2, 0).astype(int)

    def _random_input_CIFAR(self):
        """
        Returns random network input to be optimized
        :return: random input
        """
        img = np.random.uniform(0, 255, (1, 3, 224, 224))
        img /= 255.
        img -= mean_CIFAR[None, :, None, None]
        img /= std_CIFAR[None, :, None, None]
        return torch.FloatTensor(img).requires_grad_(True)

    def _input2image_CIFAR(self, img):
        # dereference batch dim
        img = img[0].data.clone().cpu().numpy()
        img *= std_CIFAR[:, None, None]
        img += mean_CIFAR[:, None, None]
        img *= 255.
        img[np.where(img > 255)] = 255
        return img.transpose(1, 2, 0).astype(int)

    def get_output(self, inp):
        for i, layer in enumerate(self.net.features):
            inp = layer(inp)
            if i == self.layer_idx:
                break
        return inp

    def get_input_map(self):
        inp = self._random_input_CIFAR() if self.dataset == 'CIFAR' else self._random_input_MNIST()
        optim = torch.optim.SGD([inp], lr=self.lr, weight_decay=1e-4)
        for i in range(self.num_updates):
            optim.zero_grad()
            activation = self.get_output(inp)
            loss = -torch.mean(activation)
            loss.backward()
            optim.step()
        image = self._input2image_CIFAR(inp) if self.dataset == 'CIFAR' else self._input2image_MNIST(inp)
        return image


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--lr', type=float, default=10, required=False)
    parser.add_argument('--num_updates', type=int, default=500, required=False)
    args = parser.parse_args()

    net = alexnet(pretrained=True)

    input_map = InputMap(net, 12, 0, lr=args.lr, num_updates=args.num_updates, dataset='CIFAR')
    img = input_map.get_input_map()

    plt.imshow(img)