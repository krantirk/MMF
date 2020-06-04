# Copyright (c) Facebook, Inc. and its affiliates.

import weakref
from abc import ABC
from typing import List, Optional

from mmf.trainers.hooks.base import HookBase
from mmf.trainers.hooks.checkpoint import CheckpointHook
from mmf.trainers.hooks.early_stopping import EarlyStoppingHook
from mmf.trainers.hooks.logistics import LogisticsHook
from mmf.trainers.hooks.lr_scheduler import LRSchedulerHook


class TrainerHooksMixin(ABC):
    def register_hooks(self, hooks: List[Optional[HookBase]]):
        """
        Register hooks to the trainer. The hooks are executed in the order
        they are registered.
        Args:
            hooks (list[Optional[HookBase]]): list of hooks
        """
        hooks = [h for h in hooks if h is not None]
        for h in hooks:
            assert isinstance(h, HookBase)
            h.trainer = weakref.proxy(self)
        self._hooks.extend(hooks)

    def configure_hooks(self):
        self.checkpoint_hook = CheckpointHook(self.config, self)
        self.early_stop_hook = EarlyStoppingHook(self.config, self)
        self.logistics_hook = LogisticsHook(self.config, self)
        self.lr_scheduler_hook = LRSchedulerHook(self.config, self)

        self.register_hooks(
            [
                self.checkpoint_hook,
                self.early_stop_hook,
                self.logistics_hook,
                self.lr_scheduler_hook,
            ]
        )

    def on_init(self):
        """Called when the trainer initialization begins, model has not yet been set."""
        for hook in self._hooks:
            hook.on_init(self)

    def before_train(self, **kwargs):
        """Called when training begins."""
        for hook in self._hooks:
            hook.before_train(**kwargs)

    def after_train(self, **kwargs):
        """Called when training ends."""
        for hook in self._hooks:
            hook.after_train(**kwargs)

    def before_train_step(self, **kwargs):
        """Called when the training batch begins."""
        for hook in self._hooks:
            hook.before_train_step(**kwargs)

    def after_train_step(self, **kwargs):
        """Called when the training batch ends."""
        for hook in self._hooks:
            hook.after_train_steps(**kwargs)

    def before_validation(self, **kwargs):
        """Called when the validation loop begins."""
        for hook in self._hooks:
            hook.before_validation(**kwargs)

    def after_validation(self, **kwargs):
        """Called when the validation loop ends."""
        for hook in self._hooks:
            hook.after_validation(**kwargs)

    def before_validation_step(self, **kwargs):
        """Called when the validation batch begins."""
        for hook in self._hooks:
            hook.before_validation_step(**kwargs)

    def after_validation_step(self, **kwargs):
        """Called when the validation batch ends."""
        for hook in self._hooks:
            hook.after_validation_step(**kwargs)

    def before_inference(self, **kwargs):
        """Called when the test begins."""
        for hook in self._hooks:
            hook.before_inference(**kwargs)

    def after_inference(self, **kwargs):
        """Called when the test ends."""
        for hook in self._hooks:
            hook.after_inference(**kwargs)

    def before_inference_step(self, **kwargs):
        """Called when the test batch begins."""
        for hook in self._hooks:
            hook.before_inference_step(**kwargs)

    def after_inference_step(self, **kwargs):
        """Called when the test batch ends."""
        for hook in self._hooks:
            hook.after_inference_step(**kwargs)
