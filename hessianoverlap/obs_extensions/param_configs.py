#!/usr/bin/env python
# -*-coding:utf-8-*-


"""
"""


# ##############################################################################
# # PARAMS
# ##############################################################################
class ObsParamConfigs:
    """
    This static class contains a semantic description of the DL architectures
    used in the DeepOBS problems, together with convenience functionality to
    filter parameter groups by semantics and retrieve their indexes.
    :cvar PARAM_CONFIGS: Dictionary in the form ``obs_task: description``,
       where ``description`` has one entry per weight (or bias) in the order
       that they are returned by each ``model.parameters()``. Each entry has
       3 elements: ``layer_type, param_type, shape``, where param_type can be
       ``weight`` or ``bias``.
    """

    PARAM_CONFIGS = {
        # (layer type, weight/bias, shape)
        "quadratic_deep": [
            ("linear", "weight", (100, 100)),
            ("linear", "bias", (100,)),
            ("linear", "weight", (100, 100)),
        ],
        #
        "mnist_logreg": [
            ("linear", "weight", (10, 784)),
            ("linear", "bias", (10,)),
        ],
        "mnist_mlp": [
            ("linear", "weight", (1000, 784)),
            ("linear", "bias", (1000,)),
            ("linear", "weight", (500, 1000)),
            ("linear", "bias", (500,)),
            ("linear", "weight", (100, 500)),
            ("linear", "bias", (100,)),
            ("linear", "weight", (10, 100)),
            ("linear", "bias", (10,)),
        ],
        "mnist_2c2d": [
            ("conv2d", "weight", (32, 1, 5, 5)),
            ("conv2d", "bias", (32,)),
            ("conv2d", "weight", (64, 32, 5, 5)),
            ("conv2d", "bias", (64,)),
            ("linear", "weight", (1024, 3136)),
            ("linear", "bias", (1024,)),
            ("linear", "weight", (10, 1024)),
            ("linear", "bias", (10,)),
        ],
        "mnist_vae": [
            ("conv2d", "weight", (64, 1, 4, 4)),
            ("conv2d", "bias", (64,)),
            ("conv2d", "weight", (64, 64, 4, 4)),
            ("conv2d", "bias", (64,)),
            ("conv2d", "weight", (64, 64, 4, 4)),
            ("conv2d", "bias", (64,)),
            ("linear", "weight", (8, 3136)),
            ("linear", "bias", (8,)),
            ("linear", "weight", (8, 3136)),
            ("linear", "bias", (8,)),
            #
            ("linear", "weight", (24, 8)),
            ("linear", "bias", (24,)),
            ("linear", "weight", (49, 24)),
            ("linear", "bias", (49,)),
            #
            ("deconv2d", "weight", (1, 64, 4, 4)),
            ("deconv2d", "bias", (64,)),
            ("deconv2d", "weight", (64, 64, 4, 4)),
            ("deconv2d", "bias", (64,)),
            ("deconv2d", "weight", (64, 64, 4, 4)),
            ("deconv2d", "bias", (64,)),
            ("linear", "weight", (784, 12544)),
            ("linear", "bias", (784,)),
        ],
        #
        "fmnist_logreg": [
            ("linear", "weight", (10, 784)),
            ("linear", "bias", (10,)),
        ],
        "fmnist_mlp": [
            ("linear", "weight", (1000, 784)),
            ("linear", "bias", (1000,)),
            ("linear", "weight", (500, 1000)),
            ("linear", "bias", (500,)),
            ("linear", "weight", (100, 500)),
            ("linear", "bias", (100,)),
            ("linear", "weight", (10, 100)),
            ("linear", "bias", (10,)),
        ],
        "fmnist_2c2d": [
            ("conv2d", "weight", (32, 1, 5, 5)),
            ("conv2d", "bias", (32,)),
            ("conv2d", "weight", (64, 32, 5, 5)),
            ("conv2d", "bias", (64,)),
            ("linear", "weight", (1024, 3136)),
            ("linear", "bias", (1024,)),
            ("linear", "weight", (10, 1024)),
            ("linear", "bias", (10,)),
        ],
        "fmnist_vae": [
            ("conv2d", "weight", (64, 1, 4, 4)),
            ("conv2d", "bias", (64,)),
            ("conv2d", "weight", (64, 64, 4, 4)),
            ("conv2d", "bias", (64,)),
            ("conv2d", "weight", (64, 64, 4, 4)),
            ("conv2d", "bias", (64,)),
            ("linear", "weight", (8, 3136)),
            ("linear", "bias", (8,)),
            ("linear", "weight", (8, 3136)),
            ("linear", "bias", (8,)),
            #
            ("linear", "weight", (24, 8)),
            ("linear", "bias", (24,)),
            ("linear", "weight", (49, 24)),
            ("linear", "bias", (49,)),
            #
            ("deconv2d", "weight", (1, 64, 4, 4)),
            ("deconv2d", "bias", (64,)),
            ("deconv2d", "weight", (64, 64, 4, 4)),
            ("deconv2d", "bias", (64,)),
            ("deconv2d", "weight", (64, 64, 4, 4)),
            ("deconv2d", "bias", (64,)),
            ("linear", "weight", (784, 12544)),
            ("linear", "bias", (784,)),
        ],
        #
        "cifar10_3c3d": [
            ("conv2d", "weight", (64, 3, 5, 5)),
            ("conv2d", "bias", (64,)),
            ("conv2d", "weight", (96, 64, 3, 3)),
            ("conv2d", "bias", (96,)),
            ("conv2d", "weight", (128, 96, 3, 3)),
            ("conv2d", "bias", (128,)),
            #
            ("linear", "weight", (512, 1152)),
            ("linear", "bias", (512,)),
            ("linear", "weight", (256, 512)),
            ("linear", "bias", (256,)),
            ("linear", "weight", (10, 256)),
            ("linear", "bias", (10,)),
        ],
        #
        "cifar100_3c3d": [
            ("conv2d", "weight", (64, 3, 5, 5)),
            ("conv2d", "bias", (64,)),
            ("conv2d", "weight", (96, 64, 3, 3)),
            ("conv2d", "bias", (96,)),
            ("conv2d", "weight", (128, 96, 3, 3)),
            ("conv2d", "bias", (128,)),
            #
            ("linear", "weight", (512, 1152)),
            ("linear", "bias", (512,)),
            ("linear", "weight", (256, 512)),
            ("linear", "bias", (256,)),
            ("linear", "weight", (100, 256)),
            ("linear", "bias", (100,)),  # 100
        ],
        "cifar100_allcnnc": [
            ("conv2d", "weight", (96, 3, 3, 3)),
            ("conv2d", "bias", (96,)),
            ("conv2d", "weight", (96, 96, 3, 3)),
            ("conv2d", "bias", (96,)),
            ("conv2d", "weight", (96, 96, 3, 3)),
            ("conv2d", "bias", (96,)),
            #
            ("conv2d", "weight", (192, 96, 3, 3)),
            ("conv2d", "bias", (192,)),
            ("conv2d", "weight", (192, 192, 3, 3)),
            ("conv2d", "bias", (192,)),
            ("conv2d", "weight", (192, 192, 3, 3)),
            ("conv2d", "bias", (192,)),
            ("conv2d", "weight", (192, 192, 3, 3)),
            ("conv2d", "bias", (192,)),
            ("conv2d", "weight", (192, 192, 1, 1)),
            ("conv2d", "bias", (192,)),
            ("conv2d", "weight", (100, 192, 1, 1)),
            ("conv2d", "bias", (100,)),
        ],
        #
        "svhn_3c3d": [
            ("conv2d", "weight", (64, 3, 5, 5)),
            ("conv2d", "bias", (64,)),
            ("conv2d", "weight", (96, 64, 3, 3)),
            ("conv2d", "bias", (96,)),
            ("conv2d", "weight", (128, 96, 3, 3)),
            ("conv2d", "bias", (128,)),
            #
            ("linear", "weight", (512, 1152)),
            ("linear", "bias", (512,)),
            ("linear", "weight", (256, 512)),
            ("linear", "bias", (256,)),
            ("linear", "weight", (10, 256)),
            ("linear", "bias", (10,)),
        ],
        "svhn_wrn164": [
            ("conv2d", "weight", (16, 3, 3, 3)),
            # block 11
            ("batchnorm2d", "weight", (16)),
            ("batchnorm2d", "bias", (16)),
            ("conv2d", "weight", (64, 16, 1, 1)),  # skip conv for upchan
            ("conv2d", "weight", (64, 16, 3, 3)),  # first conv in block
            ("batchnorm2d", "weight", (64)),
            ("batchnorm2d", "bias", (64)),
            ("conv2d", "weight", (64, 64, 3, 3)),  # last conv in block
            # block 12 (no upchan skipconv needed here)
            ("batchnorm2d", "weight", (64)),
            ("batchnorm2d", "bias", (64)),
            ("conv2d", "weight", (64, 64, 3, 3)),
            ("batchnorm2d", "weight", (64)),
            ("batchnorm2d", "bias", (64)),
            ("conv2d", "weight", (64, 64, 3, 3)),
            # block 21
            ("batchnorm2d", "weight", (64)),
            ("batchnorm2d", "bias", (64)),
            ("conv2d", "weight", (128, 64, 1, 1)),  # skip conv for upchan
            ("conv2d", "weight", (128, 64, 3, 3)),
            ("batchnorm2d", "weight", (128)),
            ("batchnorm2d", "bias", (128)),
            ("conv2d", "weight", (128, 128, 3, 3)),
            # block 22 (no upchan skipconv needed here)
            ("batchnorm2d", "weight", (128)),
            ("batchnorm2d", "bias", (128)),
            ("conv2d", "weight", (128, 128, 3, 3)),
            ("batchnorm2d", "weight", (128)),
            ("batchnorm2d", "bias", (128)),
            ("conv2d", "weight", (128, 128, 3, 3)),
            # block 31
            ("batchnorm2d", "weight", (128)),
            ("batchnorm2d", "bias", (128)),
            ("conv2d", "weight", (256, 128, 1, 1)),  # skip conv for upchan
            ("conv2d", "weight", (256, 128, 3, 3)),
            ("batchnorm2d", "weight", (256)),
            ("batchnorm2d", "bias", (256)),
            ("conv2d", "weight", (256, 256, 3, 3)),
            # block 32 (no upchan skipconv needed here)
            ("batchnorm2d", "weight", (256)),
            ("batchnorm2d", "bias", (256)),
            ("conv2d", "weight", (256, 256, 3, 3)),
            ("batchnorm2d", "weight", (256)),
            ("batchnorm2d", "bias", (256)),
            ("conv2d", "weight", (256, 256, 3, 3)),
            #
            ("batchnorm2d", "weight", (256)),
            ("batchnorm2d", "bias", (256)),
            ("linear", "weight", (10, 256)),
            ("linear", "bias", (10,)),
        ],
    }

    @classmethod
    def get_param_idxs(cls, obs_problem, layers=None, types=None, shape_fn=None):
        """
        Retrieve indexes for a given task that satisfy given criteria.

        :param obs_problem: A string like ``mnist_mlp``.
        :param layers: A list of desired layers, e.g. ``['linear', 'conv']``.
        :param types: A list of desired types, e.g. ``['weight', 'bias']``
        :param shape_fn: Function in the form ``fn(shape): bool``, returns True
          if such a shape is desired.

        Usage example: Get only small biases:
        idxs = get_obs_param_idxs("fmnist_mlp", types=["bias"],
                                  shape_fn=lambda shape: shape[0] < 500)
        """
        if layers is None:
            layer_fn = lambda detail: True
        else:
            layer_fn = lambda detail: detail[0] in layers
        #
        if types is None:
            type_fn = lambda detail: True
        else:
            assert all(
                t in {"weight", "bias"} for t in types
            ), "Types must be either weight or bias!"
            type_fn = lambda detail: detail[1] in types
        #
        if shape_fn is None:
            new_shape_fn = lambda detail: True
        else:
            new_shape_fn = lambda detail: shape_fn(detail[2])
        #
        details = cls.PARAM_CONFIGS[obs_problem]
        idxs = [
            i
            for i, d in enumerate(details)
            if layer_fn(d) and type_fn(d) and new_shape_fn(d)
        ]
        return idxs
