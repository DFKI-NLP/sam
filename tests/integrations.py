import json
import os
import sys
import time
from typing import List

import pytest
from allennlp import commands

train_path = 'data/dummy_corpus/general/train'
val_path = 'data/dummy_corpus/general/dev'
test_path = 'data/dummy_corpus/general/test'


def execute_command(args: List):
    sys.argv = args
    start = time.time()
    commands.main()
    end = time.time()
    print(f"Total training time is {end - start}")


def initialize_test(task: str, tmp_path):
    """
    In order to initiate test, there are several parameters which needs to be defined and some of them depends upon
    type (adu or rel).
    :param task: string that specify type, which can be either adu or rel
    :return: a dictionary containing metadata required for testing
    """
    metadata = {'model_path': str(tmp_path / f'{task}/sanity_test')}
    os.makedirs(metadata['model_path'], exist_ok=True)
    metadata['config_path'] = f'allennlp_configs/{task}_best.jsonnet'
    metadata['predictions_path'] = str(tmp_path / f'{task}@dummy_corpus/sanity_test/predictiononly')
    os.makedirs(metadata['predictions_path'], exist_ok=True)
    metadata['eval_path'] = str(tmp_path / f'{task}@sanity_test')
    os.makedirs(metadata['eval_path'], exist_ok=True)

    metadata['default_eval_command'] = ['allennlp', 'evaluate', '-o',
                                        '{"dataset_reader.num_shards":null,"dataset_reader.dataset_splits":{'
                                        '"test":":1"}}',
                                        '--cuda-device', '0']
    metadata['default_predict_command'] = ['allennlp', 'predagg', '--predictor', 'brat-store', '-o',
                                           '{"data_loader.shuffle":false,"dataset_reader.show_gold":false,'
                                           '"dataset_reader.show_prediction":true, '
                                           '"dataset_reader.num_shards":null,"dataset_reader.dataset_splits":{'
                                           '"test":":1"}}',
                                           '--use-dataset-reader', '--cuda-device', '0', '--silent']
    metadata['default_overrides'] = '{"trainer.num_epochs":5,"train_data_path":"data/dummy_corpus/general/train@train",' \
                                    '"validation_data_path":"data/dummy_corpus/general/dev@val",' \
                                    '"dataset_reader.dataset_splits":' \
                                    '{"train":":1","val":":1"}}'
    metadata['default_train_command'] = ['allennlp', 'train', '-o', metadata['default_overrides']]

    return metadata


@pytest.mark.slow
def test_adu(tmp_path):
    """
    Test training, prediction and evaluation pipeline for ADU identification using allennlp
    :param name: dictionary containing command line arguments
    :return: None
    """
    metadata = initialize_test('adu', tmp_path)
    args_train = metadata['default_train_command']
    args_predict = metadata['default_predict_command']
    args_eval = metadata['default_eval_command']

    assert os.path.exists(metadata['config_path']), f'Configuration file not found.'
    assert os.path.exists(metadata['model_path']), f" {metadata['model_path']}"

    args_train.extend(['-f', metadata['config_path'], '-s', metadata['model_path']])
    execute_command(args_train)

    assert os.path.exists(metadata['predictions_path']), f"{metadata['predictions_path']}"
    assert os.path.exists(test_path), f" {test_path}"

    args_predict.extend(['--batch-size', '8', '--output-file', metadata['predictions_path'],
                         metadata['model_path'], test_path + '@test'])
    execute_command(args_predict)

    assert os.path.exists(metadata['eval_path']), f" {metadata['eval_path']}"

    args_eval.extend(['--batch-size', '8', '--output-file', metadata['eval_path'] + '/metrics.json',
                      metadata['model_path'], test_path + '@test'])
    execute_command(args_eval)


@pytest.mark.slow
def test_rel(tmp_path):
    """
    Test training, prediction and evaluation pipeline for relation extraction using allennlp
    :param name: dictionary containing command line arguments
    :return: None
    """
    metadata = initialize_test('rel', tmp_path)
    args_train = metadata['default_train_command']
    args_predict = metadata['default_predict_command']
    args_eval = metadata['default_eval_command']

    assert os.path.exists(metadata['config_path']), f'Configuration file not found.'
    assert os.path.exists(metadata['model_path']), f" {metadata['model_path']} not found"

    overrides = json.loads(metadata['default_overrides'])
    overrides.update({"dataset_reader.add_negative_relations_portion": -1.0})

    args_train[args_train.index('-o') + 1] = json.dumps(overrides)
    args_train.extend(['-f', metadata['config_path'], '-s', metadata['model_path']])
    execute_command(args_train)

    overrides = json.loads(metadata['default_predict_command'][1 + metadata['default_predict_command'].index('-o')])
    overrides.update({"dataset_reader.add_negative_relations_portion": -1.0})

    args_predict[1 + metadata['default_predict_command'].index('-o')] = json.dumps(overrides)
    args_predict.extend(
        ['--batch-size', '128', '--output-file', metadata['predictions_path'],
         metadata['model_path'], test_path + '@test'])

    assert os.path.exists(metadata['predictions_path']), f"{metadata['predictions_path']} not found"
    assert os.path.exists(test_path), f" {test_path} not found"
    execute_command(args_predict)

    overrides = json.loads(metadata['default_eval_command'][1 + metadata['default_eval_command'].index('-o')])
    overrides.update({"dataset_reader.add_negative_relations_portion": -1.0})

    args_eval[1 + metadata['default_eval_command'].index('-o')] = json.dumps(overrides)
    args_eval.extend(['--batch-size', '128', '--output-file', metadata['eval_path'] + '/metrics.json',
                      metadata['model_path'], test_path + '@test'])

    assert os.path.exists(metadata['eval_path']), f" {metadata['eval_path']} not found"
    execute_command(args_eval)
