# Copyright (c) Facebook, Inc. and its affiliates.

from mmf.trainers.hooks.base import HookBase
from mmf.utils.build import build_scheduler


class LRSchedulerHook(HookBase):
    """A hook which executes a torch builtin LR scheduler and summarizes the LR.
    It is executed after every iteration.
    """

    def __init__(self, config, trainer):
        """
        Args:
            scheduler (torch.optim._LRScheduler)
        """
        super().__init__(config, trainer)
        self._scheduler = None

        if self.training_config.lr_scheduler is True:
            self._scheduler = build_scheduler(self.trainer.optimizer, self.config)

    def after_train_step(self):
        if self._scheduler is not None:
            self._scheduler.step()
