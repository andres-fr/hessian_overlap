#!/usr/bin/env python
# -*-coding:utf-8-*-


"""
"""


from collections import OrderedDict
import torch

from torchvision.io import read_image
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models import resnet50, ResNet50_Weights


# ##############################################################################
# #
# ##############################################################################
class FlexiMLP(torch.nn.Sequential):
    """ """

    def __init__(
        self, dims=(256, 20, 20, 20, 20, 20, 10), activation=torch.nn.Tanh
    ):
        """
        Defaults from the K-FAC paper
        """
        layers = []
        layers.append(("flatten", torch.nn.Flatten()))
        for i, (in_d, out_d) in enumerate(zip(dims[:-2], dims[1:-1]), 1):
            layers.append((f"linear{i}", torch.nn.Linear(in_d, out_d)))
            layers.append((f"activation{i}", activation()))
        layers.append((f"linear{i+1}", torch.nn.Linear(dims[-2], dims[-1])))
        #
        super().__init__(OrderedDict(layers))

        self.initialize()

    def initialize(self):
        """
        The mnist_mlp OBS initialization doesn't seem to do any good here, so
        we translate the default PyTorch initialization from
        https://github.com/pytorch/pytorch/blob/799521fae5e883c6d6a2beec24c9004afbc9170d/torch/nn/modules/linear.py#L103
        """
        for module in self.modules():
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.kaiming_uniform_(module.weight, a=(5**0.5))
                if module.bias is not None:
                    fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(
                        module.weight
                    )
                    bound = (fan_in**-0.5) if fan_in > 0 else 0
                    torch.nn.init.uniform_(module.bias, -bound, bound)


# ##############################################################################
# #
# ##############################################################################
def ResNet18(pretrained=False):
    """ """
    if pretrained:
        weights = ResNet18_Weights.DEFAULT
    else:
        weights = None
    return resnet18(weights=weights)


def ResNet50(pretrained=False):
    """ """
    if pretrained:
        weights = ResNet50_Weights.DEFAULT
    else:
        weights = None
    return resnet50(weights=weights)
