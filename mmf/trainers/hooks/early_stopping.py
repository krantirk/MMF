# Copyright (c) Facebook, Inc. and its affiliates.

from mmf.trainers.hooks.base import HookBase
from mmf.utils.distributed import broadcast_scalar
from mmf.utils.early_stopping import EarlyStopping


class EarlyStoppingHook(HookBase):
    """A hook which executes Early Stopping mechanism and checks if it training
    should continue or stop.
    """

    def __init__(self, config, trainer):
        """
        Args:
            optimizer (torch.optim.Optimizer):
            scheduler (torch.optim._LRScheduler)
        """
        super().__init__(config, trainer)

        early_stop_criteria = self.training_config.early_stop.criteria
        early_stop_minimize = self.training_config.early_stop.minimize
        early_stop_enabled = self.training_config.early_stop.enabled
        early_stop_patience = self.training_config.early_stop.patience
        self.early_stopping = EarlyStopping(
            self.trainer.model,
            self.trainer.checkpoint_hook.checkpoint,
            early_stop_criteria,
            patience=early_stop_patience,
            minimize=early_stop_minimize,
            should_stop=early_stop_enabled,
        )

    def after_validation(self):
        stop = self.early_stopping(
            self.trainer.num_updates, self.trainer.current_iteration, self.trainer.meter
        )
        stop = bool(broadcast_scalar(stop, src=0, device=self.trainer.device))
        return stop
