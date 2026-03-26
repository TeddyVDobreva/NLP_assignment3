from pathlib import Path
from typing import Any
import numpy as np

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
)

LABELS = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}


def plot_confusion_matrix(model: Any, dataset: pd.DataFrame, dataset_name: str) -> None:
    """
    Function evaluates the classification and saves its confusion matrix.

    Arguments:
        y_true: Series- True labels.
        y_predict: Series- Predicted labels.
        model_name: str- Name of the model.
        dataset: str- Dataset identifier.

    Returns: None
    """
    results = model.predict(dataset)
    y_predict = [np.argmax(a) for a in results[0]]
    y_true = dataset["label"]
    Path("plots").mkdir(exist_ok=True)
    print(classification_report(y_true, y_predict))

    cm = confusion_matrix(y_true, y_predict)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(xticks_rotation="vertical")

    plt.savefig(f"plots/confusion_matrix_{dataset_name}")
    plt.close()


def get_misclassified_examples(
    model,
    dataset,
) -> None:
    """
    Collects misclassified examples from the test set

    Args:
        model: model used for the prediction
        dataset: the dataset to find the mistakes on

    Returns:
        None
    """
    model.evaluate()
    results = model.predict(dataset)
    predictions = [np.argmax(a) for a in results[0]]
    errs = []
    for ex, pred in zip(dataset, predictions):
        y = int(ex["label"])
        if pred != y:
            snippet = ex["text"].replace("\n", " ")
            errs.append((y, pred, snippet))

    _show_errors(errs)


def _show_errors(errs: list) -> None:
    """
    Prints the misclassified examples

    Args:
        name: name of the model
        errs: list containing the true label, predicted label, and a part of the text
    """
    errs = sorted(errs, key=lambda x: x[0], reverse=True)
    errs = sorted(errs, key=lambda x: x[1], reverse=True)
    for i, (y, p, snip) in enumerate(errs):
        print()
        print(f"error {i + 1}")
        print("true:", LABELS[y], "pred:", LABELS[p])
        print("text:", snip)
        # snip = snip.replace("&", "\\&").replace("\\", "$\\setminus$")
        # print(f"{i + 1} & {LABELS[p]} & {LABELS[y]} & {snip}\\\\")
        # print("\\hline")
