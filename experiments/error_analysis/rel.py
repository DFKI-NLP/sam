import matplotlib.pyplot as plt
import pandas as pd


def get_error_type(true_label, prediction):
    if true_label == prediction:
        return "true positive"
    elif true_label == "unassignable":
        return "false positive"
    elif prediction == "undetected":
        return "false negative"
    else:
        raise ValueError("unknown combination")


def plot(grouped, column):
    grouped_data = (grouped[column].value_counts() / grouped.apply(len))
    for key in grouped.groups:
        values = grouped_data.xs(key=key)
        # exclude min value
        values = values[values > min(values)]
        values.plot.bar()
        plt.title(key)
        plt.suptitle(column)
        plt.xticks(rotation=45, ha="right")
        plt.show()


def plot_sub(grouped, column, sort_by_index=False, file_name=None, horizontal=True):
    grouped_data = (grouped[column].value_counts() / grouped.apply(len))
    df_keys = pd.DataFrame(grouped.groups.keys())
    s = 1.5
    if horizontal:
        x_keys = df_keys[0].unique()
        y_keys = df_keys[1].unique()
        figsize=(6.4 * s, 4.8 * s)
    else:
        x_keys = df_keys[1].unique()
        y_keys = df_keys[0].unique()
        figsize = (4.8 * s, 6.4 * s)
    fig, ax = plt.subplots(len(x_keys), len(y_keys), sharey=True, figsize=figsize)
    print(x_keys)
    print(y_keys)
    #plt.suptitle(column)
    for i, x in enumerate(x_keys):
        for j, y in enumerate(y_keys):
            values: pd.Series = grouped_data.xs(key=(x, y) if horizontal else (y, x))
            if sort_by_index:
                values = values.sort_index()
            # exclude min value
            if len(values) > 5:
                values = values[values > min(values)]
            _ax = ax[i][j]
            values.plot.bar(ax=_ax)
            if j == 0:
                _ax.set_ylabel(x)
            if i == 0:
                _ax.set_title(y)
            _ax.set_xlabel(None)
            if max([len(idx) for idx in values.index]) > 3:
                _ax.set_xticklabels(values.index, rotation=45, ha="right")
            else:
                _ax.set_xticklabels(values.index, rotation=0, ha="center")
    fig.tight_layout()
    if file_name is not None:
        plt.savefig(file_name, bbox_inches='tight', pad_inches=0.05)
    plt.show()


if __name__ == "__main__":
    # exported from as csv from https://docs.google.com/spreadsheets/d/1waetUcADyhc9hD_yTqpegablK7aZcYAfYwINwDluGNI/edit#gid=0
    df = pd.read_csv("experiments/error_analysis/sci-arg - rel - error analysis.tsv", sep="\t")

    # rearrange: relation, error_type
    df["relation"] = df.apply(
        lambda row: row["prediction"] if row["true_label"] == "unassignable" else row["true_label"], axis=1)
    df["error_type"] = df.apply(lambda row: get_error_type(row["true_label"], row["prediction"]), axis=1)
    # normalize columns
    df["arguments"] = df["arguments"].apply(lambda args: "\n".join(sorted(args.split(","))))
    df["same sentence"] = df["same sentence"].replace({"x": "yes", "o": "no"})
    df["connector"] = df["connector_claims"].fillna("NONE").apply(lambda c: c.split("+")[0].replace("SQUARE", ""))
    df["data and claim arguments in same sentence"] = df.apply(lambda row: "yes" if ("claim" in row["arguments"] and "data" in row["arguments"] and row["same sentence"] == "yes") else "no", axis=1)

    grouped = df.groupby(["relation", "error_type"])

    cols_of_interest = ["connector", "same sentence", "arguments", "data and claim arguments in same sentence"]
    for col in cols_of_interest:
        # to print full result:
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            print(grouped[col].value_counts())

    # relative
    for col in cols_of_interest:
        # to print full result:
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            print(grouped[col].value_counts() / grouped.apply(len))

    # plot
    horizontal = False
    base_fn = "experiments/error_analysis/errors_rel"
    if horizontal:
        base_fn += "_hor"
    for col in cols_of_interest:
        sort_by_index = col != "connector"
        plot_sub(grouped, col, file_name=f"{base_fn}_{col.replace(' ', '_')}.png" if base_fn else None, horizontal=horizontal, sort_by_index=sort_by_index)
