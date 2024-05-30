#!/usr/bin/env python
# -*-coding:utf-8-*-


"""
"""


import torch
import deepobs

#
from .datasets import mnist_16x16, cifar10det, cifar100det, imagenetdet
from .models import FlexiMLP, ResNet18, ResNet50

from .datasets import fake_imagenetdet


# ##############################################################################
# # MNIST MINI
# ##############################################################################
class mnist_mini(deepobs.pytorch.testproblems.mnist_mlp):
    """
    Like superclass, but uses a mini-MLP and downsampled MNIST dataset
    """

    MLP_DIMS = (256, 20, 20, 20, 20, 20, 10)
    MLP_ACTIVATION = torch.nn.Tanh

    def __init__(self, batch_size, l2_reg=None):
        """ """
        super().__init__(batch_size, l2_reg)

    def set_up(self, random_seed=0):
        """ """
        self.data = mnist_16x16(self._batch_size, random_seed=random_seed)
        self.loss_function = torch.nn.CrossEntropyLoss
        self.net = FlexiMLP(dims=self.MLP_DIMS, activation=self.MLP_ACTIVATION)
        self.net.to(self._device)
        self.regularization_groups = self.get_regularization_groups()


# ##############################################################################
# # CIFAR10_3C3D WITH DETERMINISTIC AUGMENTATIONS
# ##############################################################################
class cifar10det_3c3d(deepobs.pytorch.testproblems.cifar10_3c3d):
    """
    Uses the deterministic dataset.
    """

    def __init__(self, batch_size, l2_reg=0.1):
        """ """
        super().__init__(batch_size, l2_reg)

    def set_up(self, random_seed=0):
        """ """
        self.data = cifar10det(self._batch_size, random_seed=random_seed)
        self.loss_function = torch.nn.CrossEntropyLoss
        self.net = (
            deepobs.pytorch.testproblems.testproblems_modules.net_cifar10_3c3d(
                num_outputs=10
            )
        )
        self.net.to(self._device)
        self.regularization_groups = self.get_regularization_groups()


# ##############################################################################
# # CIFAR100_ALLCNNC WITH DETERMINISTIC AUGMENTATIONS
# ##############################################################################
class cifar100det_allcnnc(deepobs.pytorch.testproblems.cifar100_allcnnc):
    """
    Uses the deterministic dataset.
    """

    def __init__(self, batch_size, l2_reg=0.0005):
        """ """
        super().__init__(batch_size, l2_reg)

    def set_up(self, random_seed=0):
        """Set up the All CNN C test problem on Cifar-100."""
        self.data = cifar100det(self._batch_size, random_seed=random_seed)
        self.loss_function = torch.nn.CrossEntropyLoss
        self.net = (
            deepobs.pytorch.testproblems.testproblems_modules.net_cifar100_allcnnc()
        )
        self.net.to(self._device)
        self.regularization_groups = self.get_regularization_groups()


# ##############################################################################
# # CIFAR100_ALLCNNC WITH DETERMINISTIC AUGMENTATIONS
# ##############################################################################
class imagenetdet_resnet18(deepobs.pytorch.testproblems.imagenet_vgg16):
    """ """

    # if pretrained is false, kaiming normal init is used
    # https://github.com/pytorch/vision/blob/947ae1dc71867f28021d5bc0ff3a19c249236e2a/torchvision/models/resnet.py#L208
    PRETRAINED = False

    def __init__(self, batch_size, l2_reg=0.0005):
        """ """
        super().__init__(batch_size, l2_reg)

    def set_up(self, random_seed=0):
        """ """
        self.data = imagenetdet(self._batch_size, random_seed=random_seed)
        # self.data = fake_imagenetdet(
        #     self._batch_size, train_eval_size=1000, random_seed=random_seed
        # )
        self.loss_function = torch.nn.CrossEntropyLoss
        self.net = ResNet18(pretrained=self.PRETRAINED)
        self.net.to(self._device)
        self.regularization_groups = self.get_regularization_groups()


class imagenetdet_resnet50(deepobs.pytorch.testproblems.imagenet_vgg16):
    """ """

    # if pretrained is false, kaiming normal init is used
    # https://github.com/pytorch/vision/blob/947ae1dc71867f28021d5bc0ff3a19c249236e2a/torchvision/models/resnet.py#L208
    PRETRAINED = False

    def __init__(self, batch_size, l2_reg=0.0005):
        """ """
        super().__init__(batch_size, l2_reg)

    def set_up(self, random_seed=0):
        """ """
        # self.data = imagenetdet(self._batch_size, random_seed=random_seed)
        self.data = fake_imagenetdet(
            self._batch_size, train_eval_size=1000, random_seed=random_seed
        )
        self.loss_function = torch.nn.CrossEntropyLoss
        self.net = ResNet50(pretrained=self.PRETRAINED)
        self.net.to(self._device)
        self.regularization_groups = self.get_regularization_groups()
