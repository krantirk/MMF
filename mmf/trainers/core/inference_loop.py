# Copyright (c) Facebook, Inc. and its affiliates.

from abc import ABC
from typing import Any, Dict, Tuple, Type

import torch
import tqdm

from mmf.common.meter import Meter
from mmf.common.report import Report
from mmf.utils.distributed import is_master


class TrainerInferenceLoopMixin(ABC):
    def evaluate(
        self, loader, use_tqdm: bool = False, single_batch: bool = False
    ) -> Tuple[Dict[str, Any], Type[Meter]]:
        meter = Meter()

        with torch.no_grad():
            self.model.eval()
            disable_tqdm = not use_tqdm or not is_master()
            combined_report = None

            for batch in tqdm.tqdm(loader, disable=disable_tqdm):
                report = self._forward(batch)
                self._update_meter(report, meter)

                # accumulate necessary params for metric calculation
                if combined_report is None:
                    combined_report = report
                else:
                    combined_report.accumulate_tensor_fields(
                        report, self.metrics.required_params
                    )
                    combined_report.batch_size += report.batch_size

                if single_batch is True:
                    break

            combined_report.metrics = self.metrics(combined_report, combined_report)
            self._update_meter(combined_report, meter, eval_mode=True)

            # enable train mode again
            self.model.train()

        return combined_report, meter

    def inference_loop(self, dataset_type: str) -> None:
        if self.config.evaluation.predict:
            self._predict(dataset_type)
            return

        self.writer.write(f"Starting inference on {dataset_type} set")

        report, meter = self.evaluate(
            getattr(self, f"{dataset_type}_loader"), use_tqdm=True
        )
        self.after_inference(report=report, meter=meter)
        # prefix = "{}: full {}".format(report.dataset_name, dataset_type)
        # self._summarize_report(meter, prefix)

    def _predict(self, dataset_type: str) -> None:
        reporter = self.dataset_loader.get_test_reporter(dataset_type)
        with torch.no_grad():
            self.model.eval()
            message = f"Starting {dataset_type} inference predictions"
            self.writer.write(message)

            while reporter.next_dataset():
                dataloader = reporter.get_dataloader()

                for batch in tqdm(dataloader):
                    prepared_batch = reporter.prepare_batch(batch)
                    model_output = self.model(prepared_batch)
                    report = Report(prepared_batch, model_output)
                    reporter.add_to_report(report, self.model)

            self.writer.write("Finished predicting")
            self.model.train()
