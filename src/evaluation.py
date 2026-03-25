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


def plot_confusion_matrix(
    model: Any, dataset: pd.DataFrame, dataset_name: str
) -> None:
    """
    Function evaluates the classification and saves its confusion matrix.

    Arguments:
        y_true: Series- True labels.
        y_predict: Series- Predicted labels.
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
