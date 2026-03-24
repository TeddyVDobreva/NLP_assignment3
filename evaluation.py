from pathlib import Path
from typing import Any

import matplotlib as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
)


def plot_confusion_matrix(
    model: Any, loader: Any, model_name: str, dataset: str
) -> None:
    """
    Function evaluates the classification and saves its confusion matrix.

    Arguments:
        y_true: Series- True labels.
        y_predict: Series- Predicted labels.
        model_name: str- Name of the model.
        dataset: str- Dataset identifier.

    Returns: None
    """
    results = model.evaluate(loader)
    y_true = results["y_true"]
    y_predict = results["y_pred"]
    Path("plots").mkdir(exist_ok=True)
    print(classification_report(y_true, y_predict))

    cm = confusion_matrix(y_true, y_predict)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(xticks_rotation="vertical")

    plt.savefig(f"plots/confusion_matrix_{model_name}_{dataset}")
    plt.close()
