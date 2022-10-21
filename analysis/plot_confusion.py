import json

import argparse
import os.path

from analysis.calculate_metric import plot_confusion_matrix

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--json',
        help='json file path to load confusion data from',
        type=str, required=True
    )
    parser.add_argument(
        '--indices',
        type=str,
        default=None
    )

    args = parser.parse_args()
    with open(args.json) as f:
        json_data = json.load(f)
        dir_name = os.path.dirname(args.json)
        file_name = os.path.splitext(os.path.basename(args.json))[0]
        plot_confusion_matrix(
            type_matrix=json_data, filename=os.path.join(dir_name, file_name),
            special_idx='UNDETECTED', special_column='UNASSIGNABLE',
            indices_sorted=args.indices.split(",") if args.indices is not None else None
        )
