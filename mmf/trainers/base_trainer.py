# Copyright (c) Facebook, Inc. and its affiliates.

from abc import abstractmethod

from mmf.common import typings as mmf_typings
from mmf.common.registry import registry
from mmf.trainers.core.device import TrainerDeviceMixin
from mmf.trainers.core.hooks import TrainerHooksMixin
from mmf.trainers.core.inference_loop import TrainerInferenceLoopMixin
from mmf.trainers.core.profiling import TrainerProfilingMixin
from mmf.trainers.core.reporting import TrainerReportingMixin
from mmf.trainers.core.training_loop import TrainerTrainingLoopMixin
from mmf.utils.general import print_model_parameters
from mmf.utils.logger import Logger
from mmf.utils.timer import Timer


@registry.register_trainer("base_trainer")
class BaseTrainer(
    TrainerHooksMixin,
    TrainerTrainingLoopMixin,
    TrainerDeviceMixin,
    TrainerInferenceLoopMixin,
    TrainerReportingMixin,
    TrainerProfilingMixin,
):
    def __init__(self, config: mmf_typings.DictConfig):
        self.config = config
        self.training_config = self.config.training
        self.profiler = Timer()
        self.total_timer = Timer()
        self._hooks = []

    def load(self):
        self.run_type = self.config.get("run_type", "train")
        # Check if loader is already defined, else init it
        writer = registry.get("writer", no_warning=True)
        if writer:
            self.writer = writer
        else:
            self.writer = Logger(self.config)
            registry.register("writer", self.writer)

        configuration = registry.get("configuration", no_warning=True)
        if configuration:
            configuration.pretty_print()

        self.configure_device()
        self.configure_seed()

        self.load_datasets()
        self.load_model()
        self.load_optimizer()

        # Hooks Initialize
        self.configure_hooks()

        self.parallelize_model()
        self.init_meter()
        self.load_metrics()
        self.init_timers()

    @abstractmethod
    def load_datasets(self):
        """Warning: this is just empty shell for code implemented in other class."""

    @abstractmethod
    def load_model(self):
        """Warning: this is just empty shell for code implemented in other class."""

    @abstractmethod
    def load_optimizer(self):
        """Warning: this is just empty shell for code implemented in other class."""

    def train(self):
        self.writer.write("===== Model =====")
        self.writer.write(self.model)
        print_model_parameters(self.model)

        if "train" not in self.run_type:
            self.inference()
            return

        self.before_train()
        self.training_loop()
        self.after_train()

        self.inference()

    def inference(self):
        self.before_inference()
        if "val" in self.run_type:
            self.inference_loop("val")

        if any(rt in self.run_type for rt in ["inference", "test", "predict"]):
            self.inference_loop("test")
        self.after_inference()
