#!/usr/bin/env python
# -*-coding:utf-8-*-


"""
This module hosts a sampler to obtain reproducible and balanced subsets of a
PyTorch dataset.
"""


import random
from collections import defaultdict

#
import torch
from torch.utils.data import Sampler


# ##############################################################################
# # SAMPLERS
# ##############################################################################
class SubsetSampler(Sampler):
    """
    Like ``torch.utils.data.SubsetRandomsampler``, but without the random,
    and with the possibility of balanced sampling. Samples a subset of the
    given dataset from a given list of indices, without replacement.
    Usage example::

      sampler = SubsetSampler.get_balanced(mnist_dataset, size=200)
      dl = DataLoader(train_ds, batch_size=1, sampler=sampler)
      _, labels = zip(*dl)
      testdict = defaultdict(int)
      for lbl in labels:
          testdict[lbl.item()] += 1
      print(len(labels))  # should be 200
      print(testdict)  # should be {0:20, 1:20, 2:20, ...}
    """

    def __init__(self, indices):
        """
        :param indices: Integer indices for the sampling.
        """
        self.indices = indices

    def __len__(self):
        """ """
        return len(self.indices)

    def __iter__(self):
        """ """
        for idx in self.indices:
            yield idx

    @staticmethod
    def randperm_generator(n, seed=154764013):
        """
        :param seed: A large prime integer
        :yields: A pseudo-random (NOT cryptographically safe) permutation of
          the integers between 0 and ``n-1`` (included), without repetition.
          Leverages multiplicative groups, analogous to drawing a 5-point star.
        """
        seed = (n - 1) * seed
        for i in range(seed, n + seed):
            yield (i * seed) % n

    @classmethod
    def get_balanced(cls, dataset, size=100, random=False):
        """
        given a ``dataset`` that yields ``dataset[idx] = (data, label)``,
        where labels are hashable, this method returns a ``SubsetSampler``
        with ``size`` indexes, such that they are balanced among classes.
        This requires that the dataset can be integer-divided by ``size``
        across the number of its classes, and that each class has at least
        ``size / num_classes`` elements. Indexes are retrieved in ascending
        order. If ``random`` is false, the lowest indexes for each class will
        be gathered, making it deterministic.
        """
        assert len(dataset) >= size, "Size can't be larger than dataset!"
        # group all data indexes by label, and optionally shuffle them
        histogram = defaultdict(list)
        # for idx, (_, lbl) in enumerate(dataset):
        for idx, lbl in enumerate(dataset.targets):
            # if lbl is a tensor, try to convert to native, otherwise
            # two tensors with same val still map to different keys
            try:
                lbl = lbl.item()
            except AttributeError:
                pass
            histogram[lbl].append(idx)
        if random:
            for v in histogram.values():
                random.shuffle(v)
        # sanity check:
        class_sizes = {k: len(v) for k, v in histogram.items()}
        num_classes = len(histogram)
        entries_per_class, rest = divmod(size, num_classes)
        assert rest == 0, "Please choose a size divisible by num_classes!"
        assert all(
            v >= entries_per_class for v in class_sizes.values()
        ), f"Not all classes have enough elements! {class_sizes}"
        # now we can gather the balanced indexes into the sampler
        idxs = sorted(
            sum((v[:entries_per_class] for v in histogram.values()), [])
        )
        sampler = cls(idxs)
        return sampler


class SeededRandomSampler(Sampler):
    """
    Similar to ``torch.utils.data.Randomsampler``, but with a specific
    seeding protocol: Given indices and a seed at initialization, it generates
    a permutation of the given indices. Then, every time it is called again,
    (e.g. next epoch), generates a permutation with ``seed+1``.
    This ensures that, given an initial seed, subsequent behaviour is
    reproducible.

      sampler = SeededRandomSampler(list(len(train_ds)), initial_seed=0)
      dl = DataLoader(train_ds, batch_size=1, sampler=sampler)
    """

    # generator seeds are encouraged to have a good balance between zeros and
    # ones. one way to ensure this is to add a large, balanced number to all
    # seeds
    MIN_SEED = 4933327

    @staticmethod
    def randperm(num_els=5, seed=0):
        """
        :yields: Entries from an integer tensor containing a permutation of
          numbers from 0 to ``num_els-1`` (both included), that have been
          permuted using the given ``seed``.
        """
        generator = torch.Generator()
        generator.manual_seed(seed)
        yield from torch.randperm(num_els, generator=generator).tolist()

    def __init__(self, indices, initial_seed=0):
        """
        :param indices: Collection of integer indices for the sampling.
        """
        self.indices = list(indices)
        self.initial_seed = initial_seed
        self.set_seed(initial_seed)

    def __len__(self):
        """ """
        return len(self.indices)

    def __iter__(self):
        """ """
        for p in self.randperm(len(self), self.seed):
            yield self.indices[p]
        # increment seed at the end of generator (e.g. epoch)
        self.seed += 1

    def set_seed(self, seed):
        """ """
        self.seed = self.initial_seed + self.MIN_SEED
