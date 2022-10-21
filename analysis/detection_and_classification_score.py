import argparse
import json

import pandas as pd


def get_scores_from_counts(tp, fp, fn):
    p = tp / (tp + fp)
    r = tp / (tp + fn)
    f1 = (2 * p * r) / (p + r)
    return {"p": p, "r": r, "f1": f1}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--json',
        help='json file path to load confusion data from',
        type=str,
        required=True,
    )
    parser.add_argument(
        '--json_out',
        help='evaluation results output json file',
        type=str,
        default=None,
    )

    args = parser.parse_args()
    eval_results = {'span_detection': dict(), 'span_classification': dict()}
    with open(args.json) as f:
        json_data = json.load(f)

    confusion_matrix = pd.DataFrame.from_records(json_data).fillna(0).astype(int)

    UNDETECTED = "UNDETECTED"
    UNASSIGNABLE = "UNASSIGNABLE"

    classes = [c for c in confusion_matrix.columns if c != UNASSIGNABLE]

    # calculate classification performance
    confusion_matrix_classification = confusion_matrix.loc[classes, classes]
    classification_counts_per_class = {"tp": pd.Series({c: confusion_matrix.loc[c, c] for c in classes})}
    classification_counts_per_class["fp"] = confusion_matrix_classification.sum(axis="columns") - classification_counts_per_class["tp"]
    classification_counts_per_class["fn"] = confusion_matrix_classification.sum(axis="index") - classification_counts_per_class["tp"]
    scores_classification_per_class = get_scores_from_counts(**classification_counts_per_class)

    # calculate detection performance
    counts_detection = {
        "tp": confusion_matrix_classification.sum().sum(),
        "fn": confusion_matrix.loc[UNDETECTED].sum(),
        "fp": confusion_matrix.loc[:, UNASSIGNABLE].sum(),
    }
    scores_detection = get_scores_from_counts(**counts_detection)

    results = {
        "detection": scores_detection,
        "classification": {k: v.mean() for k, v in scores_classification_per_class.items()}
    }

    if args.json_out is not None:
        print(f"write output to: {args.json_out}")
        with open(args.json_out, 'w') as f:
            json.dump(results, f, indent=2)
    else:
        print(json.dumps(results, indent=2))
