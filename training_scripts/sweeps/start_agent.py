import wandb
import os
import argparse


def run(count=1):
    """
    It starts agent for a sweep using sweep id obtained from environment variable.
    :param count: number of trials to runs
    :return: None
    """

    sweep_id = os.environ['WANDB_NOTES']
    wandb.agent(sweep_id, count=count)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Experiment Args')
    parser.add_argument(
        '--count', '-c',
        help='number of runs for wandb agent',
        type=int, required=True
    )
    args, _ = parser.parse_known_args()
    run(count=args.count)
