from typing import Callable

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def evaluate_intents(
    dataframe: pd.DataFrame,
    predictor: Callable[[str], str],
) -> dict:
    if dataframe.empty:
        return {}

    y_true = dataframe["intent"].astype(str).tolist()
    y_pred = [
        predictor(message)
        for message in dataframe["message"].astype(str)
    ]

    labels = sorted(set(y_true) | set(y_pred))

    return {
        "accuracy": round(
            accuracy_score(y_true, y_pred),
            4,
        ),
        "macro_precision": round(
            precision_score(
                y_true,
                y_pred,
                labels=labels,
                average="macro",
                zero_division=0,
            ),
            4,
        ),
        "macro_recall": round(
            recall_score(
                y_true,
                y_pred,
                labels=labels,
                average="macro",
                zero_division=0,
            ),
            4,
        ),
        "macro_f1": round(
            f1_score(
                y_true,
                y_pred,
                labels=labels,
                average="macro",
                zero_division=0,
            ),
            4,
        ),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=labels,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(
            y_true,
            y_pred,
            labels=labels,
        ).tolist(),
        "labels": labels,
    }


def evaluate_escalation(
    dataframe: pd.DataFrame,
    predictor: Callable[[str], bool],
) -> dict:
    if dataframe.empty:
        return {}

    y_true = (
        dataframe["should_escalate"]
        .astype(bool)
        .tolist()
    )

    y_pred = [
        bool(predictor(message))
        for message in dataframe["message"].astype(str)
    ]

    return {
        "accuracy": round(
            accuracy_score(y_true, y_pred),
            4,
        ),
        "precision": round(
            precision_score(
                y_true,
                y_pred,
                zero_division=0,
            ),
            4,
        ),
        "recall": round(
            recall_score(
                y_true,
                y_pred,
                zero_division=0,
            ),
            4,
        ),
        "f1": round(
            f1_score(
                y_true,
                y_pred,
                zero_division=0,
            ),
            4,
        ),
    }


def simple_majority_baseline(dataframe: pd.DataFrame) -> dict:
    """
    Majority-class baseline for intent classification.
    """
    if dataframe.empty:
        return {}

    majority_intent = (
        dataframe["intent"]
        .value_counts()
        .index[0]
    )

    y_true = dataframe["intent"].tolist()
    y_pred = [majority_intent] * len(y_true)

    return {
        "baseline": "majority_class",
        "majority_intent": majority_intent,
        "accuracy": round(
            accuracy_score(y_true, y_pred),
            4,
        ),
        "macro_f1": round(
            f1_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            ),
            4,
        ),
    }


def keyword_baseline(dataframe: pd.DataFrame) -> dict:
    """
    Simple baseline: classifies based on a few manually selected words.
    """
    def predict(text: str) -> str:
        text = text.lower()

        if "refund" in text:
            return "refund_request"

        if "late" in text or "delivery" in text:
            return "delivery_delay"

        if "password" in text or "login" in text:
            return "account_access"

        if "payment" in text or "charged" in text:
            return "payment_problem"

        return "other"

    if dataframe.empty:
        return {}

    y_true = dataframe["intent"].tolist()
    y_pred = [
        predict(message)
        for message in dataframe["message"]
    ]

    return {
        "baseline": "simple_keyword_baseline",
        "accuracy": round(
            accuracy_score(y_true, y_pred),
            4,
        ),
        "macro_f1": round(
            f1_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            ),
            4,
        ),
    }
