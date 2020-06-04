# Copyright (c) Facebook, Inc. and its affiliates.

import omegaconf

from mmf.common.dataset_loader import DatasetLoader
from mmf.common.registry import registry
from mmf.trainers.base_trainer import BaseTrainer
from mmf.utils.build import build_model, build_optimizer


@registry.register_trainer("mmf_trainer")
class MMFTrainer(BaseTrainer):
    def __init__(self, config):
        super().__init__(config)

    def load_datasets(self):
        self.writer.write("Loading datasets", "info")
        self.dataset_loader = DatasetLoader(self.config)
        self.dataset_loader.load_datasets()

        self.train_dataset = self.dataset_loader.train_dataset
        self.val_dataset = self.dataset_loader.val_dataset
        self.test_dataset = self.dataset_loader.test_dataset

        self.train_loader = self.dataset_loader.train_loader
        self.val_loader = self.dataset_loader.val_loader
        self.test_loader = self.dataset_loader.test_loader

    def load_model(self):
        self.writer.write("Loading model", "info")
        attributes = self.config.model_config[self.config.model]
        # Easy way to point to config for other model
        if isinstance(attributes, str):
            attributes = self.config.model_config[attributes]

        with omegaconf.open_dict(attributes):
            attributes.model = self.config.model

        self.model = build_model(attributes)
        self.model = self.model.to(self.device)

    def load_optimizer(self):
        self.writer.write("Loading optimizer", "info")
        self.optimizer = build_optimizer(self.model, self.config)
