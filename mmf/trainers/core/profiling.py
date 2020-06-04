# Copyright (c) Facebook, Inc. and its affiliates.

from abc import ABC

from mmf.utils.timer import Timer


class TrainerProfilingMixin(ABC):
    def init_timers(self):
        self.profiler = Timer()
        self.not_debug = self.training_config.logger_level != "debug"

    def profile(self, text):
        if self.not_debug:
            return
        self.writer.write(text + ": " + self.profiler.get_time_since_start(), "debug")
        self.profiler.reset()
