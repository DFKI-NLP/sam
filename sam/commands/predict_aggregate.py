"""
Similar to the predict subcommand, but allows to aggregate result from multiple instances
before writing to file or stdout.
"""

import argparse
import json
import sys
from typing import Optional

from allennlp.commands.predict import _PredictManager, Predict, _get_predictor
from allennlp.common.util import lazy_groups_of
from overrides import overrides

from allennlp.commands.subcommand import Subcommand
from allennlp.common import logging as common_logging
from tqdm import tqdm


@Subcommand.register("predagg")
class PredictAggregate(Predict):
    @overrides
    def add_subparser(self, parser: argparse._SubParsersAction) -> argparse.ArgumentParser:

        subparser = super().add_subparser(parser)
        subparser.add_argument(
            "--show-progress", action="store_true", help="show progress bar"
        )
        subparser.set_defaults(func=_predict_and_aggregate)

        return subparser


class _PredictManagerAggregate(_PredictManager):
    def __init__(
        self,
        *args,
        output_file: Optional[str],
        show_progress: bool = False,
        **kwargs
    ) -> None:
        super().__init__(*args, output_file=None, **kwargs)
        self._output_path = output_file
        self._show_progress = show_progress

    @overrides
    def run(self) -> None:
        has_reader = self._dataset_reader is not None
        index = 0
        predictions = []
        if has_reader:
            all_batches = lazy_groups_of(self._get_instance_data(), self._batch_size)
            if self._show_progress:
                assert not self._print_to_console, f'can not show progress bar if printing prediction output to console'
                all_batches = tuple(all_batches)
                all_batches = tqdm(all_batches, total=len(all_batches))
            for batch in all_batches:
                for model_input_instance, result in zip(batch, self._predict_instances(batch)):
                    self._maybe_print_to_console_and_file(index, result, str(model_input_instance))
                    index = index + 1
                    predictions.append(result)
        else:
            all_batches = lazy_groups_of(self._get_json_data(), self._batch_size)
            if self._show_progress:
                assert not self._print_to_console, f'can not show progress bar if printing prediction output to console'
                all_batches = tuple(all_batches)
                all_batches = tqdm(all_batches, total=len(all_batches))
            for batch_json in all_batches:
                for model_input_json, result in zip(batch_json, self._predict_json(batch_json)):
                    self._maybe_print_to_console_and_file(
                        index, result, json.dumps(model_input_json)
                    )
                    index = index + 1
                    predictions.append(result)

        if self._output_path is not None:
            self._predictor.write_predictions_to_file(self._output_path, predictions)


def _predict_and_aggregate(args: argparse.Namespace) -> None:
    common_logging.FILE_FRIENDLY_LOGGING = args.file_friendly_logging

    predictor = _get_predictor(args)

    if args.silent and not args.output_file:
        print("--silent specified without --output-file.")
        print("Exiting early because no output will be created.")
        sys.exit(0)

    manager = _PredictManagerAggregate(
        predictor=predictor,
        input_file=args.input_file,
        output_file=args.output_file,
        batch_size=args.batch_size,
        print_to_console=not args.silent,
        has_dataset_reader=args.use_dataset_reader,
        show_progress=args.show_progress,
    )
    manager.run()
