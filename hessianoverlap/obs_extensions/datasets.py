#!/usr/bin/env python
# -*-coding:utf-8-*-


"""
"""

import os

import torchvision

#
from deepobs import config
from deepobs.pytorch.datasets import mnist, cifar10, cifar100, imagenet
from deepobs.pytorch.datasets.cifar10 import (
    training_transform_not_augmented as cifar10_transform,
)
from deepobs.pytorch.datasets.cifar100 import (
    training_transform_not_augmented as cifar100_transform,
)
from deepobs.pytorch.datasets.imagenet import (
    training_transform_not_augmented as imagenet_transform,
)

#
from .samplers import SubsetSampler, SeededRandomSampler


# ##############################################################################
# # RESAMPLED MNIST
# ##############################################################################
class mnist_resampled(mnist):
    """
    Like the superclass, but the dataloader adds a transform to reduce size.
    """

    SIZE = None

    def __init__(self, batch_size, train_eval_size=10000, random_seed=0):
        """ """
        self.random_seed = random_seed
        super().__init__(batch_size, train_eval_size)

    def get_transform(self):
        """ """
        if self.SIZE is None:
            t = torchvision.transforms.ToTensor()
        else:
            t = torchvision.transforms.Compose(
                [
                    torchvision.transforms.Resize(self.SIZE),
                    torchvision.transforms.ToTensor(),
                ]
            )
        return t

    def _make_test_dataloader(self):
        """ """
        transform = self.get_transform()
        ds = torchvision.datasets.MNIST(
            root=config.get_data_dir(),
            train=False,
            download=True,
            transform=transform,
        )
        dl = self._make_dataloader(ds, sampler=None, shuffle=False)
        return dl

    def _make_train_eval_dataloader(self):
        """ """
        # Replace the truncated, implicit-seed sampler with our balanced,
        # reproducible one. Otherwise, evaluating at different times messes the
        # seed and also kills reproducibility during training.
        transform = self.get_transform()
        ds = torchvision.datasets.MNIST(
            root=config.get_data_dir(),
            train=True,
            download=True,
            transform=transform,
        )
        #
        sampler = SubsetSampler.get_balanced(
            self._train_dataloader.dataset,
            size=self._train_eval_size,
            random=False,
        )
        #
        dl = self._make_dataloader(ds, sampler=sampler, shuffle=False)
        return dl

    def _make_train_and_valid_dataloader(self):
        """ """
        transform = self.get_transform()
        ds = torchvision.datasets.MNIST(
            root=config.get_data_dir(),
            train=True,
            download=True,
            transform=transform,
        )
        #
        # given a seed, split into training and xv subsets.
        # note that self._train_eval_size is the size of the val, not the
        # train set (maybe was a bug in OBS?).
        indices = list(SeededRandomSampler.randperm(len(ds), self.random_seed))
        valid_indices, train_indices = (
            indices[: self._train_eval_size],
            indices[self._train_eval_size :],
        )
        #
        train_sampler = SeededRandomSampler(train_indices, self.random_seed)
        valid_sampler = SeededRandomSampler(valid_indices, self.random_seed)
        # since random sampling, shuffle is useless
        train_loader = self._make_dataloader(ds, sampler=train_sampler)
        valid_loader = self._make_dataloader(ds, sampler=valid_sampler)
        return train_loader, valid_loader


class mnist_16x16(mnist_resampled):
    """ """

    SIZE = 16


# ##############################################################################
# # CIFAR 10 WITHOUT TRAINING DATA AUGMENTATION
# ##############################################################################
class cifar10det(cifar10):
    """
    Deterministic: no training data augmentation.
    Plus all the reproducible sampler extensions.
    """

    def __init__(
        self,
        batch_size,
        data_augmentation=False,
        train_eval_size=10000,
        random_seed=0,
    ):
        """ """
        self.random_seed = random_seed
        super().__init__(batch_size, False, train_eval_size)

    def _make_test_dataloader(self):
        transform = cifar10_transform
        test_dataset = torchvision.datasets.CIFAR10(
            root=config.get_data_dir(),
            train=False,
            download=True,
            transform=transform,
        )
        return self._make_dataloader(test_dataset, sampler=None, shuffle=False)

    def _make_train_eval_dataloader(self):
        """ """
        ds = self._train_dataloader.dataset
        sampler = SubsetSampler.get_balanced(
            ds, size=self._train_eval_size, random=False
        )
        dl = self._make_dataloader(ds, sampler=sampler, shuffle=False)
        return dl

    def _make_train_and_valid_dataloader(self):
        """ """
        # always not augmented
        transform = cifar10_transform

        ds = torchvision.datasets.CIFAR10(
            root=config.get_data_dir(),
            train=True,
            download=True,
            transform=transform,
        )
        # given a seed, split into training and xv subsets.
        # note that self._train_eval_size is the size of the val, not the
        # train set (maybe was a bug in OBS?).
        indices = list(SeededRandomSampler.randperm(len(ds), self.random_seed))
        valid_indices, train_indices = (
            indices[: self._train_eval_size],
            indices[self._train_eval_size :],
        )
        #
        train_sampler = SeededRandomSampler(train_indices, self.random_seed)
        valid_sampler = SeededRandomSampler(valid_indices, self.random_seed)
        # since random sampling, shuffle is useless
        train_loader = self._make_dataloader(ds, sampler=train_sampler)
        valid_loader = self._make_dataloader(ds, sampler=valid_sampler)
        return train_loader, valid_loader


# ##############################################################################
# # CIFAR 100 WITH DETERMINISTIC AUGMENTATIONS
# ##############################################################################
class cifar100det(cifar100):
    """
    Deterministic: no training data augmentation
    """

    def __init__(
        self,
        batch_size,
        data_augmentation=True,
        train_eval_size=10000,
        random_seed=0,
    ):
        """ """
        self.random_seed = random_seed
        super().__init__(batch_size, False, train_eval_size)

    def _make_test_dataloader(self):
        transform = cifar100_transform
        test_dataset = torchvision.datasets.CIFAR100(
            root=config.get_data_dir(),
            train=False,
            download=True,
            transform=transform,
        )
        return self._make_dataloader(test_dataset, sampler=None, shuffle=False)

    def _make_train_eval_dataloader(self):
        """ """
        ds = self._train_dataloader.dataset
        sampler = SubsetSampler.get_balanced(
            ds, size=self._train_eval_size, random=False
        )
        dl = self._make_dataloader(ds, sampler=sampler, shuffle=False)
        return dl

    def _make_train_and_valid_dataloader(self):
        """ """
        # always not augmented
        transform = cifar100_transform

        ds = torchvision.datasets.CIFAR100(
            root=config.get_data_dir(),
            train=True,
            download=True,
            transform=transform,
        )
        # given a seed, split into training and xv subsets.
        # note that self._train_eval_size is the size of the val, not the
        # train set (maybe was a bug in OBS?).
        indices = list(SeededRandomSampler.randperm(len(ds), self.random_seed))
        valid_indices, train_indices = (
            indices[: self._train_eval_size],
            indices[self._train_eval_size :],
        )
        #
        train_sampler = SeededRandomSampler(train_indices, self.random_seed)
        valid_sampler = SeededRandomSampler(valid_indices, self.random_seed)
        # since random sampling, shuffle is useless
        train_loader = self._make_dataloader(ds, sampler=train_sampler)
        valid_loader = self._make_dataloader(ds, sampler=valid_sampler)
        return train_loader, valid_loader


# ##############################################################################
# # IMAGENET WITH DETERMINISTIC AUGMENTATIONS
# ##############################################################################
class imagenetdet(imagenet):
    """
    Deterministic: no training data augmentation
    """

    def __init__(
        self,
        batch_size,
        data_augmentation=True,
        train_eval_size=50000,
        random_seed=0,
    ):
        """ """
        self.random_seed = random_seed
        super().__init__(batch_size, False, train_eval_size)

    def _make_test_dataloader(self):
        transform = imagenet_transform
        test_dataset = torchvision.datasets.ImageNet(
            root=os.path.join(config.get_data_dir(), "imagenet", "pytorch"),
            split="val",
            transform=transform,
        )
        return self._make_dataloader(test_dataset, sampler=None, shuffle=False)

    def _make_train_eval_dataloader(self):
        """ """
        ds = self._train_dataloader.dataset
        sampler = SubsetSampler.get_balanced(
            ds, size=self._train_eval_size, random=False
        )
        dl = self._make_dataloader(ds, sampler=sampler, shuffle=False)
        return dl

    def _make_train_and_valid_dataloader(self):
        """ """
        # always not augmented
        transform = imagenet_transform
        #
        ds = torchvision.datasets.ImageNet(
            root=os.path.join(config.get_data_dir(), "imagenet", "pytorch"),
            split="train",
            transform=transform,
        )
        # given a seed, split into training and xv subsets.
        indices = list(SeededRandomSampler.randperm(len(ds), self.random_seed))
        valid_indices, train_indices = (
            indices[: self._train_eval_size],
            indices[self._train_eval_size :],
        )
        #
        train_sampler = SeededRandomSampler(train_indices, self.random_seed)
        valid_sampler = SeededRandomSampler(valid_indices, self.random_seed)
        # since random sampling, shuffle is useless
        train_loader = self._make_dataloader(ds, sampler=train_sampler)
        valid_loader = self._make_dataloader(ds, sampler=valid_sampler)
        return train_loader, valid_loader


class FakeDataIdx(torchvision.datasets.FakeData):
    """ """

    def __getitem__(self, idx):
        """ """
        x, y = super().__getitem__(idx)
        y = idx % self.num_classes
        return x, y


class fake_imagenetdet(imagenetdet):
    """ """

    TRAIN_SIZE = 2000  # 1281167
    TEST_SIZE = 2000  # 50000

    def _make_test_dataloader(self):
        """ """
        test_dataset = FakeDataIdx(
            self.TEST_SIZE,
            (3, 224, 224),
            1000,
            torchvision.transforms.ToTensor(),
        )

        return self._make_dataloader(test_dataset, sampler=None, shuffle=False)

    def _make_train_eval_dataloader(self):
        """ """
        ds = self._train_dataloader.dataset
        sampler = SubsetSampler.get_balanced(
            ds, size=self._train_eval_size, random=False
        )
        dl = self._make_dataloader(ds, sampler=sampler, shuffle=False)
        return dl

    def _make_train_and_valid_dataloader(self):
        """ """
        ds = FakeDataIdx(
            self.TRAIN_SIZE,
            (3, 224, 224),
            1000,
            torchvision.transforms.ToTensor(),
        )
        # given a seed, split into training and xv subsets.
        indices = list(SeededRandomSampler.randperm(len(ds), self.random_seed))
        valid_indices, train_indices = (
            indices[: self._train_eval_size],
            indices[self._train_eval_size :],
        )
        #
        train_sampler = SeededRandomSampler(train_indices, self.random_seed)
        valid_sampler = SeededRandomSampler(valid_indices, self.random_seed)
        # since random sampling, shuffle is useless
        train_loader = self._make_dataloader(ds, sampler=train_sampler)
        valid_loader = self._make_dataloader(ds, sampler=valid_sampler)
        return train_loader, valid_loader
