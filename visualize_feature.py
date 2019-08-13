import torch
import numpy as np
from PIL import Image


def input2gray(img):
    img -= img.min()
    img *= 255. / (img.max() - img.min())
    return Image.fromarray(img.astype(int), 'L')


def input2image(img):
    if len(img.shape) > 2:
        axis = (1, 2)
        mode = 'RGBA'
    else:
        axis = (0, 1)
        mode = 'L'
    img -= img.min(axis=axis)
    img *= 255. / img.max(axis=axis)
    return Image.fromarray(img.astype(int), mode)


class FeatureTracker:
    """
    Class to handle visualization of features in a trained network by optimizing input to maximize their activation
    """

    def __init__(self, network, mod_name, feat_idx):
        self.net = network
        self.module = network._modules[mod_name]
        self.feat_idx = feat_idx
        self.hook = self.module.register_forward_hook(self.make_f_hook())
        self.feat_out = None

    def make_f_hook(self):
        def f_hook(mod, inp, out):
            self.feat_out = out[:, self.feat_idx]

        return f_hook

    def get_output(self, inp):
        self.net(inp)
        return self.feat_out

    def remove_hook(self):
        self.hook.remove_hook()


class InputOptimizer:
    """
    Class to control optimization of input for maximization of given feature activation
    """

    def __init__(self, feature_tracker, shape=(1, 1, 28, 28), px_min=-0.4242107, px_max=1.352058):
        self.feature_tracker = feature_tracker
        self.input_shape = shape
        self.px_min = px_min
        self.px_max = px_max

    def _random_input(self):
        """
        Returns random network input to be optimized
        :return: random input
        """
        return torch.FloatTensor(np.random.uniform(self.px_min, self.px_max, self.input_shape)).requires_grad_(True)

    def optimize(self, num_updates, lr):
        inp = self._random_input()
        for i in range(num_updates):
            out = self.feature_tracker.get_output(inp)
            torch.sum(out).backward()
            inp.data += inp.grad.data * lr
            inp.grad.zero_()
        # return optimized input
        return inp.data.squeeze().cpu().numpy()


def visualize_feats(network, *features, num_updates=2000, lr=0.005, **input_args):
    images = []
    for feature in features:
        # get FeatureTracker object
        feat = FeatureTracker(network, *feature)
        # get InputOptimizer
        optim = InputOptimizer(feat, **input_args)
        inp = optim.optimize(num_updates, lr)
        # return the image
        images += input2image(inp)
    return images

