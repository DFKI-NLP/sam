import argparse
from collections import defaultdict
from os import listdir, path
from os.path import isfile, join

from analysis.calculate_metric import get_metrics

ADU_ARGS = dict(path_gold='experiments/prediction/adu/best_uncased_10r5ge6a_goldonly',
            path_predicted='experiments/prediction/adu/best_uncased_10r5ge6a_predictiononly',
            out_dir='', exclude_labels_with_suffix='GOLD')
BASENAME_ADU_OUTPUT = 'experiments/evaluation/using_pipeline/adu/best_uncased_10r5ge6a_per_file'
RELGOLD_ARGS = dict(path_gold='experiments/prediction/rel@gold_adus/best_uncased_257eyrv1_goldonly',
            path_predicted='experiments/prediction/rel@gold_adus/best_uncased_257eyrv1_predictiononly',
            out_dir='', exclude_labels_with_suffix=None, type_blacklist='semantically_same',
            merge_span_annotations_with_relation_label='parts_of_same')

BASENAME_RELGOLD_OUTPUT = 'experiments/evaluation/using_pipeline/rel@gold_adus/best_uncased_257eyrv1_per_file'

RELPRED_ARGS = dict(path_gold='experiments/prediction/rel@gold_adus/best_uncased_257eyrv1_goldonly',
            path_predicted='experiments/prediction/rel@predicted_adus/best_uncased_257eyrv1_predictiononly',
            out_dir='', exclude_labels_with_suffix=None, type_blacklist='semantically_same',
            merge_span_annotations_with_relation_label='parts_of_same')

BASENAME_RELPRED_OUTPUT = 'experiments/evaluation/using_pipeline/rel@predicted_adus/best_uncased_257eyrv1_per_file'

def get_files_without_part(files):
    result = defaultdict(list)
    for f in files:
        name = f.split('[')[0]
        result[name].append(f)
    return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--predicted_path',
        help='folder path for predicted data',
        type=str, required=True
    )

    parser.add_argument(
        '--type',
        help='adu, relg, relp',
        type=str, required=True
    )

    args = parser.parse_args()

    predicted_file_names = [f.split('.')[0] for f in listdir(args.predicted_path) if isfile(join(args.predicted_path, f)) and f.endswith('.ann')]

    predicted_files_without_parts = get_files_without_part(predicted_file_names)
    filenames = sorted(predicted_files_without_parts, key = lambda x:x)
    for name in filenames:
        cm_args = {}
        if args.type == 'adu':
            ADU_ARGS.update({'out_dir':path.join(BASENAME_ADU_OUTPUT,name),
                             'file_whitelist': ",".join(predicted_files_without_parts[name])})
            cm_args = ADU_ARGS
        elif args.type == 'relg':
            RELGOLD_ARGS.update({'out_dir': path.join(BASENAME_RELGOLD_OUTPUT, name),
                             'file_whitelist': ",".join(predicted_files_without_parts[name])})
            cm_args = RELGOLD_ARGS

        elif args.type == 'relp':
            RELPRED_ARGS.update({'out_dir': path.join(BASENAME_RELPRED_OUTPUT, name),
                             'file_whitelist': ",".join(predicted_files_without_parts[name])})
            cm_args = RELPRED_ARGS
        else:
            raise ValueError(f'Given type: {args.type} is invalid')

        get_metrics(**cm_args)




