#!/usr/bin/env python
import json

import argparse
import sys

import torch
import wandb
import os
import multiprocessing
import collections
import statistics

from training_scripts.sweeps.execution import integrate_arguments_and_run_allennlp

Worker = collections.namedtuple("Worker", ("queue", "process"))
WorkerInitData = collections.namedtuple(
    "WorkerInitData", ("num", "sweep_id", "sweep_run_name",
                       "serialization_dir", "num_folds")
)
WorkerDoneData = collections.namedtuple("WorkerDoneData", ("metrics"))


def ld_to_dl(ld):
    v = {k: [dic[k] for dic in ld] for k in ld[0]}
    return v


def reset_wandb_env():
    """
    This method removes all the WandB environment variable previously assigned a value except for WANDB_PROJECT,
    WANDB_ENTITY and WANDB_API_KEY.
    :return:
    """
    exclude = {
        "WANDB_PROJECT",
        "WANDB_ENTITY",
        "WANDB_API_KEY",
    }
    for k, v in os.environ.items():
        if k.startswith("WANDB_") and k not in exclude:
            del os.environ[k]


def train(sweep_q, worker_q):
    """
    This method is used when worker process is started. It fetch meta data for worker process using get() method. This
    data is then filtered to create a dictionary for wandb callback arguments and allennlp config arguments which is
    passed to another script (execution.py) that processes all these arguments and start allennlp train command. This
    method waits until training is completed and then append metrics generated by training to sweep queue.

    :param sweep_q: multiprocessing queue for worker to place output of the it's process
    :param worker_q: multiprocessing queue of the worker to get meta data for it's process
    :return: None
    """
    reset_wandb_env()
    worker_data = worker_q.get()
    serialization_dir = worker_data.serialization_dir
    num_folds = worker_data.num_folds
    fold_index = worker_data.num
    run_name = "{}-{}".format(worker_data.sweep_run_name, worker_data.num)
    run_id = wandb.util.generate_id()
    # TODO : Currently all callbacks are overwritten, this needs to be changed to integrate existing callbacks
    # We configure W&B callback which is used as a list item to trainer.callbacks in allennlp config. It contains key
    # value pairs which is passed to wandb.init(). More information on AllenNLP wandb callbacks
    #  https://docs.allennlp.org/main/api/training/callbacks/wandb/
    wandb_callback_args = dict(group=worker_data.sweep_id,
                               name=run_name,
                               wandb_kwargs=dict(
                                   job_type=worker_data.sweep_run_name,
                                   id=run_id
                               ),
                               type="custom_wandb",
                               entity=os.getenv("WANDB_ENTITY"),
                               project=os.getenv("WANDB_PROJECT"),
                               files_to_save=["config.json", "out.log", "metrics.json"]
                               )
    serialization_dir = os.path.join(serialization_dir, run_id)
    integrate_arguments_and_run_allennlp(serialization_dir=serialization_dir,
                                         overrides_default={'dataset_reader.num_shards': num_folds,
                                                            'dataset_reader.dev_shard_idx': fold_index,
                                                            'trainer.callbacks': [wandb_callback_args]})
    wandb.join()
    with open(os.path.join(serialization_dir, 'metrics.json')) as f:
        metrics = json.load(f)
    sweep_q.put(WorkerDoneData(metrics=metrics))


def main(num_folds, serialization_dir):
    """
    This method uses multiprocessing for implementing k-cross validation. It spin up k (num folds) workers before
    starting sweep by calling wandb.init(). Workers will be blocked on a queue waiting to start. Once sweep is initiated
    this method creates meta data for each worker process(wandb agent run) using sweep_run meta data and current fold
    index. Then worker process is started which starts the agent run and on completion returns metrics. Mean, median and
    standard deviation of each metric from different runs is calculated and then logged to sweep_run.

    :param num_folds: number of folds for k-cross validation
    :param serialization_dir: path to store wandb runs
    :return: None
    """

    sweep_q = multiprocessing.Queue()
    workers = []
    for num in range(num_folds):
        q = multiprocessing.Queue()
        p = multiprocessing.Process(
            target=train, kwargs=dict(sweep_q=sweep_q, worker_q=q)
        )
        p.start()
        workers.append(Worker(queue=q, process=p))
    os.makedirs(serialization_dir, exist_ok=True)
    sweep_run = wandb.init(dir=serialization_dir)
    sweep_id = sweep_run.sweep_id or "unknown"
    sweep_url = sweep_run.get_sweep_url()
    project_url = sweep_run.get_project_url()
    sweep_group_url = "{}/groups/{}".format(project_url, sweep_id)
    sweep_run.notes = sweep_group_url
    sweep_run.save()
    sweep_run_name = sweep_run.name or sweep_run.id or "unknown"

    metrics = []
    for num in range(num_folds):
        worker = workers[num]
        # start worker
        worker.queue.put(
            WorkerInitData(
                sweep_id=sweep_id,
                num=num,
                sweep_run_name=sweep_run_name,
                serialization_dir=serialization_dir,
                num_folds=num_folds
            )
        )
        # get metric from worker
        result = sweep_q.get()
        # wait for worker to finish
        worker.process.join()
        metrics.append(result.metrics)

    metrics = ld_to_dl(metrics)
    metrics_avg = {k + '_mean': statistics.mean(v) for k, v in metrics.items() if k.startswith('best')}
    metrics_stdev = {k + '_std': statistics.stdev(v) for k, v in metrics.items() if k.startswith('best')}
    metrics_median = {k + '_median': statistics.median(v) for k, v in metrics.items() if k.startswith('best')}
    # log metric to sweep_run
    sweep_run.log({**metrics_avg, **metrics_stdev, **metrics_median})
    wandb.join()

    print("*" * 40)
    print("Sweep URL:       ", sweep_url)
    print("Sweep Group URL: ", sweep_group_url)
    print("*" * 40)


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    parser = argparse.ArgumentParser(description='Experiment Args')

    parser.add_argument(
        '--num_folds',
        help='Number of cross folds',
        type=int, required=True
    )

    parser.add_argument(
        '--serialization_dir', '-s',
        help='root serialization directory',
        type=str, required=True
    )
    args, remaining_args = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining_args
    main(**vars(args))