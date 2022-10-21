import os
from typing import Dict, Any

from allennlp.training.callbacks.callback import TrainerCallback
from allennlp.training.callbacks.log_writer import LogWriterCallback
from allennlp.training.callbacks.wandb import WandBCallback
from overrides import overrides


@TrainerCallback.register("custom_wandb")
class CustomWandBCallback(WandBCallback):
    @overrides
    def on_end(
            self,
            trainer: "GradientDescentTrainer",
            metrics: Dict[str, Any] = None,
            epoch: int = None,
            is_primary: bool = True,
            **kwargs,
    ) -> None:
        self.wandb.log(metrics)
        if is_primary:
            self.close()

    @overrides
    def on_start(
        self, trainer: "GradientDescentTrainer", is_primary: bool = True, **kwargs
    ) -> None:
        # WandBCallback calls super.on_start() to initialize trainer in LogWriterCallback, but we do that by directly
        # calling on_start() on LogWriterCallback.
        LogWriterCallback.on_start(self, trainer, is_primary=is_primary)
        if not is_primary:
            return None

        import wandb

        self.wandb = wandb

        # When using CustomWandBCallback with train-cross-validation.py, we get an error that multiple run_ids created,
        # for single run. This is because we create one run_id in train-cross-validation.py and send it as kwargs to
        # callback but on_start() method of WandBCallback checks if run_id is none before updating parameters from
        # kwargs. Due to this two run_ids are being created. Therefore, we override on_start() method of WandBCallback
        # to prevent this situation by not generating any run_id in the method.

        self.wandb.init(**self._wandb_kwargs)

        for fpath in self._files_to_save:
            self.wandb.save(  # type: ignore
                os.path.join(self.serialization_dir, fpath), base_path=self.serialization_dir
            )

        if self._watch_model:
            self.wandb.watch(self.trainer.model)  # type: ignore
