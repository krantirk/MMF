# Copyright (c) Facebook, Inc. and its affiliates.

from typing import Any

from mmf.common import typings as mmf_typings


class HookBase:
    """
    Base class for hooks that can be registered with :class:`BaseTrainer`.

    Attributes:
        config: Config for the hook
        trainer: A weak reference to the trainer object. Set by the trainer when the
            hook is registered.
    """

    def __init__(self, config: mmf_typings.DictConfig, trainer: Any):
        self.config = config
        self.trainer = trainer
        self.training_config = self.config.training

    def on_init(self):
        """
        Called before the first iteration.
        """
        pass

    def before_train(self, **kwargs):
        """
        Called before training starts.
        """
        pass

    def after_train(self, **kwargs):
        """
        Called after training ends.
        """
        pass

    def before_train_step(self, **kwargs):
        """
        Called before each train iteration.
        """
        pass

    def after_train_step(self, **kwargs):
        """
        Called after each train iteration.
        """
        pass

    def before_validation(self, **kwargs):
        """
        Called before validation starts.
        """
        pass

    def after_validation(self, **kwargs):
        """
        Called after validation ends.
        """
        pass

    def before_validation_step(self, **kwargs):
        """
        Called before each validation iteration.
        """
        pass

    def after_validation_step(self, **kwargs):
        """
        Called after each validation iteration.
        """
        pass

    def before_inference(self, **kwargs):
        """
        Called before test starts.
        """
        pass

    def after_inference(self, **kwargs):
        """
        Called after test ends.
        """
        pass

    def before_inference_step(self, **kwargs):
        """
        Called before each test iteration.
        """
        pass

    def after_inference_step(self, **kwargs):
        """
        Called after each test iteration.
        """
        pass
