#!/usr/bin/env python
# -*-coding:utf-8-*-


"""
"""


import random

# import importlib
#
import numpy as np
import torch
import deepobs
from deepobs.pytorch import config, testproblems

#
from . import problems as obs_ext_problems


# ##############################################################################
# # RUNNERS
# ##############################################################################
class ObsExtPTRunner(deepobs.pytorch.runners.runner.PTRunner):
    """
    Like superclass, but create_testproblem is modified to load problems from
    our custom extension
    """

    @staticmethod
    def create_testproblem(testproblem, batch_size, l2_reg, random_seed):
        """Sets up the deepobs.pytorch.testproblems.testproblem instance.
        Args:
            testproblem (str): The name of the testproblem.
            batch_size (int): Batch size that is used for training
            l2_reg (float): Regularization factor
            random_seed (int): The random seed of the framework
        Returns:
            deepobs.pytorch.testproblems.testproblem: An instance of
            deepobs.pytorch.testproblems.testproblem
        """
        # set the seed and GPU determinism
        if config.get_is_deterministic():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        # Find testproblem by name and instantiate with batch size and L2reg
        try:
            # REPLACED THESE 2 LINES, WHICH ASSUME A DIFFERENT FILESYS STRUCT
            # testproblem_mod = importlib.import_module(testproblem)
            # testproblem_cls = getattr(testproblem_mod, testproblem)
            testproblem_cls = getattr(obs_ext_problems, testproblem)
            print("Loading local testproblem.")
        except:
            testproblem_cls = getattr(testproblems, testproblem)

        # if the user specified L2-regularization, use that one
        if l2_reg is not None:
            tproblem = testproblem_cls(batch_size, l2_reg)
        # else use the default of the testproblem
        else:
            tproblem = testproblem_cls(batch_size)

        # Set up the testproblem.
        # Our custom problems added seed-robust samplers to the dataset, which
        # expect the seed. Try to pass it, ignore otherwise
        try:
            tproblem.set_up(random_seed=random_seed)
        except TypeError:
            tproblem.set_up()
        #
        return tproblem


# ##############################################################################
# # OBS UTILS
# ##############################################################################
def get_obs_testproblem(
    tproblem_name, batch_size=32, seed=0, runner_class=ObsExtPTRunner
):
    """
    Set-up OBS problem.
    :param tproblem_name: A subset of ``cifar100_3c3d, cifar100_allcnnc,
      cifar10_3c3d, fmnist_2c2d, fmnist_logreg, fmnist_mlp, fmnist_vae,
      mnist_2c2d, mnist_logreg, mnist_mlp, mnist_vae, quadratic_deep,
      svhn_3c3d``.
      Not supported atm: ``cifar100_vgg16, cifar100_vgg19, cifar100_wrn164,
      cifar100_wrn404, cifar10_vgg16, cifar10_vgg19, imagenet_vgg16,
      imagenet_vgg19``.
    :returns: model, train loader, loss fn, eval_fn, tproblem.
    """

    # Create DeepOBS testproblem
    tproblem = runner_class.create_testproblem(
        testproblem=tproblem_name,
        batch_size=batch_size,
        l2_reg=None,
        random_seed=seed,
    )

    # Extract model, loss-function and data
    model = tproblem.net
    loss_fn = tproblem.loss_function(reduction="mean")
    losses_fn = torch.nn.CrossEntropyLoss(reduction="none")
    train_loader, _ = tproblem.data._make_train_and_valid_dataloader()

    # nested fn, Ad-hoc equivalent of PTRunner.evaluate_all
    def eval_fn():
        train_loss, train_acc = runner_class.evaluate(tproblem, phase="TRAIN")
        xv_loss, xv_acc = runner_class.evaluate(tproblem, phase="VALID")
        test_loss, test_acc = runner_class.evaluate(tproblem, phase="TEST")
        return train_loss, train_acc, xv_loss, xv_acc, test_loss, test_acc

    return model, train_loader, loss_fn, losses_fn, eval_fn, tproblem
