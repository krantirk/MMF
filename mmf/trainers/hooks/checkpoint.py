# Copyright (c) Facebook, Inc. and its affiliates.

from mmf.common.registry import registry
from mmf.trainers.hooks.base import HookBase
from mmf.utils.checkpoint import Checkpoint


class CheckpointHook(HookBase):
    """
    A hook which executes a torch builtin LR scheduler and summarizes the LR.
    It is executed after every iteration.
    """

    def __init__(self, config, trainer):
        """
        Args:
            optimizer (torch.optim.Optimizer):
            scheduler (torch.optim._LRScheduler)
        """
        super().__init__(config, trainer)
        self._checkpoint = Checkpoint(trainer)
        self._checkpoint.load_state_dict()
        self.checkpoint_interval = self.config.training.checkpoint_interval
        self.writer = registry.get("writer")

    @property
    def checkpoint(self):
        return self._checkpoint

    def after_train_step(self):
        num_updates = registry.get("num_updates")
        current_iteration = registry.get("current_iteration")
        if num_updates % self.checkpoint_interval == 0:
            self.writer.write("Checkpoint time. Saving a checkpoint.")
            self._checkpoint.save(num_updates, current_iteration, update_best=False)

    def after_train(self):
        self._checkpoint.restore()
        self._checkpoint.finalize()
