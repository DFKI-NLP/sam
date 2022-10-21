import json
import logging
import os
import shutil
from collections import defaultdict
from typing import List, cast

from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor
from overrides import overrides

from sam.dataset_reader import BratDatasetReader
from sam.dataset_reader.brat_annotation import BratAnnotationCollection

logger = logging.getLogger(__name__)


@Predictor.register("brat")
class BratPredictor(Predictor):
    """
    Predictor for any model that takes in a sentence and returns
    a single set of tags for it.  In particular, it can be used with
    the [`CrfTagger`](https://docs.allennlp.org/models/master/models/tagging/models/crf_tagger/)
    model and also the [`SimpleTagger`](../models/simple_tagger.md) model.

    Registered as a `Predictor` with name "sentence_tagger".
    """

    def __init__(
        self,
        *args,
        use_obj_store: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._json_store = {} if use_obj_store else None
        assert isinstance(self._dataset_reader, BratDatasetReader), \
            f'brat predictor requires a BratDatasetRead, but found dataset reader with type: {type(self._dataset_reader)}'
        self._dataset_reader = cast(BratDatasetReader, self._dataset_reader)

    #def predict(self, sentence: str) -> JsonDict:
    #    return self.predict_json({"sentence": sentence})

    #@overrides
    #def _json_to_instance(self, json_dict: JsonDict) -> Instance:
    #    """
    #    Expects JSON that looks like `{"sentence": "..."}`.
    #    Runs the underlying model, and adds the `"words"` to the output.
    #    """
    #    sentence = json_dict["sentence"]
    #    tokens = self._tokenizer.tokenize(sentence)
    #    return self._dataset_reader.text_to_instance(tokens)

    @overrides
    def predict_instance(self, instance: Instance) -> JsonDict:
        outputs = super().predict_instance(instance)
        return self._dataset_reader.model_output_to_json(instance, outputs, obj_store=self._json_store)

    @overrides
    def predict_batch_instance(self, instances: List[Instance]) -> List[JsonDict]:
        outputs = super().predict_batch_instance(instances)
        return [self._dataset_reader.model_output_to_json(inst, outs, obj_store=self._json_store)
                for inst, outs in zip(instances, outputs)]

    @overrides
    def dump_line(self, outputs: JsonDict) -> str:
        return json.dumps(outputs) + "\n"

    def write_predictions_to_file(self, file_path: str, predictions: List[str]):

        predictions_by_text = defaultdict(list)
        texts = {}
        for prediction_line in predictions:
            pred_dict = json.loads(prediction_line)
            if pred_dict['text_id'] in texts:
                assert texts[pred_dict['text_id']] == pred_dict['txt'], \
                    f'text does not match with previews one for text_id={pred_dict["text_id"]}'
            texts[pred_dict['text_id']] = pred_dict['txt']
            predictions_by_text[pred_dict['text_id']].extend(pred_dict['ann'])

        if os.path.exists(file_path):
            logger.warning(f'directory/file already exists, it will be removed: {file_path}')
            if os.path.isfile(file_path):
                os.remove(file_path)
            else:
                shutil.rmtree(file_path)
        os.makedirs(file_path)
        for text_id, anns in predictions_by_text.items():
            brat_annotations = BratAnnotationCollection.from_annotation_dicts(
                annotation_dicts=anns, base=texts[text_id], json_store=self._json_store)
            #brat_annotations.set_annotation_names()
            current_file_path = os.path.join(file_path, text_id)
            brat_annotations.to_files(current_file_path)
        logger.warning(f'\nProcessing Time:\n{json.dumps(self._dataset_reader._stats["time"], indent=2)}')


@Predictor.register("brat-store")
class BratPredictorWithStore(BratPredictor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, use_obj_store=True, **kwargs)
