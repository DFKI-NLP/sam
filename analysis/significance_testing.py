import argparse
import glob
import os

import numpy as np
from scipy import stats

import analysis.calculate_metric as metric_calculator


def get_test_statistics(path_gold: str, path_prediction1: str, path_prediction2: str, metric: str,
                        seed: int, size: int, num_trials: int):
    """
    This method perform bootstrap significance test and returns test statistics. It select a subset of files from set
    of files obtained from gold data. This subset of files are evaluated for both models using
    their respective predictions for num_trials times. On each trial, it saves the metric specific by user to list
    for both models. These lists are then used to perform significance test.
    :param path_gold: path to gold data
    :param path_prediction1: path to prediction files for first model
    :param path_prediction2: path to prediction files for second model
    :param metric: metric that is to be used for significance test. NOTE: it should be seperated by '/'
            eg: 'entity/macro/f1'
    :param seed: seed for randomization
    :param size: size of subset for a particular trial
    :param num_trials: total number of trials
    :return: significance test statistics i.e t-statistics and p-value
    """
    metric_model1 = []
    metric_model2 = []
    files = [os.path.basename(os.path.splitext(f)[0]) for f in glob.glob(path_gold + '/*.ann')]
    metrics = metric.split('/')

    np.random.seed(seed=seed)
    for j in range(num_trials):
        subset = np.random.choice(files, replace=True, size=size)
        file_whitelist = ','.join(subset)

        metric_model1.append(metric_calculator.get_metrics(
            path_gold=path_gold, path_predicted=path_prediction1,
            exclude_labels_with_suffix='GOLD', file_whitelist=file_whitelist)[(metrics[0], metrics[1], metrics[2])])

        metric_model2.append(metric_calculator.get_metrics(
            path_gold=path_gold, path_predicted=path_prediction2,
            exclude_labels_with_suffix='GOLD', file_whitelist=file_whitelist)[(metrics[0], metrics[1], metrics[2])])

    test_statistics = stats.ttest_ind(metric_model1, metric_model2)
    print(f'Mean for model 1: {np.mean(metric_model1)}, Mean for model 2: {np.mean(metric_model2)}')
    print(f'Standard deviation for model 1: {np.std(metric_model1)}, Standard deviation for model 2: {np.std(metric_model2)}')
    return test_statistics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Perform bootstrap significance testing between two sets of predictions from different models for '
                    'a particular metric.')

    parser.add_argument(
        '--seed',
        help='seed for randomization',
        type=int, required=True
    )

    parser.add_argument(
        '--num_trials',
        help='Number of times to repeat the experiment',
        type=int, required=True
    )

    parser.add_argument(
        '--size',
        help='Sample size for subset',
        type=int, required=True
    )

    parser.add_argument(
        '--metric',
        help='Metric used for significance testing. Metric is calculated using calculate_metric.py and output of this'
             'script saves the metric as json file. '
             'Notice that metric is represented by joining key parts using / in json file, this argument also'
             'expect input in similar fashion',
        type=str, required=True
    )

    parser.add_argument(
        '--path_gold',
        help='Path to the folder containing gold data in BRAT format',
        type=str, required=True
    )

    parser.add_argument(
        '--path_prediction1',
        help='Path to the folder containing first model predictions in BRAT format',
        type=str, required=True
    )

    parser.add_argument(
        '--path_prediction2',
        help='Path to the folder containing second model predictions in BRAT format',
        type=str, required=True
    )
    args = parser.parse_args()
    test_statistics = get_test_statistics(**vars(args))
    print(test_statistics)
