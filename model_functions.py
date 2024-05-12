import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, auc, confusion_matrix, roc_curve
from sklearn.preprocessing import MinMaxScaler, StandardScaler

NORM_COLUMNS = [
    "drive",
    "play_in_drive",
    "qtr",
    "quarter_seconds_remaining",
    "down",
    "ydstogo",
    "yardline_100",
    "season",
]
STAN_COLUMNS = ["score_differential", "spread_line"]
COLUMNS = NORM_COLUMNS + STAN_COLUMNS


def split_data(
    df, season=None, balance=False, id_info=False, normalize=False, standardize=False
):
    if id_info:  # hot-one encode coach and team info
        df = pd.get_dummies(df, columns=["posteam", "coach"])
    else:  # drop coach and team info
        df = df.drop(["coach", "posteam"], axis=1)

    # normalize and standardize
    if normalize and standardize:
        scaler = MinMaxScaler()
        for column in NORM_COLUMNS:
            if column in df.columns:
                df[column] = scaler.fit_transform(df[[column]])
        scaler = StandardScaler()
        for column in STAN_COLUMNS:
            if column in df.columns:
                df[column] = scaler.fit_transform(df[[column]])
    elif normalize:
        scaler = MinMaxScaler()
        for column in COLUMNS:
            if column in df.columns:
                df[column] = scaler.fit_transform(df[[column]])
    elif standardize:
        scaler = StandardScaler()
        for column in COLUMNS:
            if column in df.columns:
                df[column] = scaler.fit_transform(df[[column]])

    # Code run, pass to 0, 1
    df["play_type"] = df["play_type"].map({"run": 0, "pass": 1})

    # set split season to last in dataset
    if not season:
        season = max(df["season"])

    # Reserve chosen season for testing
    test_df = df[df["season"] == season]
    # Take 10% for validation
    train_df = df[df["season"] < season].sample(frac=1)
    val_df = train_df.sample(frac=0.1)
    train_df = train_df.drop(val_df.index)

    # balance the number of passes and runs in training data
    if balance:
        min_count = min(train_df["play_type"].value_counts())
        train_df = (
            train_df.groupby("play_type")
            .apply(lambda x: x.sample(n=min_count))
            .reset_index(drop=True)
        )
        train_df = train_df.sample(frac=1)

    train_X = train_df.drop("play_type", axis=1).to_numpy()
    train_y = train_df["play_type"].values
    validation_X = val_df.drop("play_type", axis=1).to_numpy()
    validation_y = val_df["play_type"].values
    test_X = test_df.drop("play_type", axis=1).to_numpy()
    test_y = test_df["play_type"].values

    training = (train_X, train_y)
    validation = (validation_X, validation_y)
    testing = (test_X, test_y)
    return training, validation, testing


def accuracies(model, training_X, training_y, validation_X, validation_y):
    train_accuracy = accuracy_score(model.predict(training_X), training_y)
    validation_accuracy = accuracy_score(model.predict(validation_X), validation_y)
    return train_accuracy, validation_accuracy


def plot_roc(model, validation_X, validation_y, color, title):
    probabilities = model.predict_proba(validation_X)[:, 1]
    fpr, tpr, _ = roc_curve(validation_y, probabilities)
    auc_score = auc(fpr, tpr)
    roc_df = pd.DataFrame({"False Positive Rate": fpr, "True Positive Rate": tpr})

    sns.lineplot(
        data=roc_df,
        x="False Positive Rate",
        y="True Positive Rate",
        linewidth=2,
        color=color,
    )
    plt.fill_between(fpr, tpr, 0, alpha=0.2, color=color)
    plt.text(0.6, 0.075, f"AUC = {auc_score:.3f}", fontsize=18)
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    if title:
        plt.title(title)
    plt.show()


def plot_confusion_matrix(model, validation_X, validation_y, color, title):
    predictions = model.predict(validation_X)
    cm = confusion_matrix(validation_y, predictions)
    sns.heatmap(
        cm,
        annot=True,
        cbar=False,
        cmap=sns.light_palette(color, as_cmap=True),
        fmt="g",
        square=True,
        xticklabels=["Run", "Pass"],
        yticklabels=["Run", "Pass"],
    )
    if title:
        plt.title(title)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.show()
