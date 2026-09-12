import argparse
from pathlib import Path

import pandas as pd

from src.agent import SupportAgent
from src.config import settings
from src.data_loader import (
    filter_brand,
    load_dataset,
    load_golden_set,
)
from src.evaluator import (
    evaluate_escalation,
    evaluate_intents,
    keyword_baseline,
    simple_majority_baseline,
)
from src.intents import classify_intent
from src.utils import save_json, save_text


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the customer-support agent pipeline."
    )

    parser.add_argument(
        "--data",
        default=settings.data_path,
        help="Path to source dataset.",
    )

    parser.add_argument(
        "--golden",
        default=settings.golden_path,
        help="Path to hand-labelled golden set.",
    )

    parser.add_argument(
        "--brand",
        default=settings.brand,
        help="Brand to evaluate.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=250,
        help="Maximum number of source rows to process.",
    )

    parser.add_argument(
        "--use-llm",
        action="store_true",
        help="Use OpenAI-compatible model when API key is available.",
    )

    return parser.parse_args()


def build_decision_log(agent: SupportAgent) -> list[dict]:
    return [
        {
            "decision": "Use transparent intent labels",
            "reason": (
                "The assignment requires explainability and a small set "
                "of intents."
            ),
        },
        {
            "decision": "Escalate payment and account issues",
            "reason": (
                "These issues may require identity verification and "
                "should not be fully automated."
            ),
        },
        {
            "decision": "Never request passwords or full card numbers",
            "reason": (
                "This reduces privacy and security risks in public support."
            ),
        },
        {
            "decision": "Use a conservative confidence threshold",
            "reason": (
                "Incorrect automatic replies can harm customer trust."
            ),
        },
        {
            "decision": "Use templates as a reliable fallback",
            "reason": (
                "The system must run even without an LLM or network access."
            ),
        },
        {
            "decision": "Use private messages for sensitive information",
            "reason": (
                "Order and account details should not be posted publicly."
            ),
        },
        {
            "decision": "Evaluate on a manually labelled golden set",
            "reason": (
                "Automatic metrics alone do not prove support quality."
            ),
        },
        {
            "decision": "Include a majority-class baseline",
            "reason": (
                "This shows whether the system learns beyond a trivial "
                "prediction."
            ),
        },
        {
            "decision": "Include a simple keyword baseline",
            "reason": (
                "This gives a lightweight comparison against the proposed "
                "classifier."
            ),
        },
        {
            "decision": "Keep escalation reasons explicit",
            "reason": (
                "Human agents should understand why automation stopped."
            ),
        },
    ]


def main():
    args = parse_args()

    print("Loading dataset...")
    dataframe = load_dataset(args.data)

    print(f"Loaded {len(dataframe)} rows.")

    brand_dataframe = filter_brand(
        dataframe,
        args.brand,
    )

    print(
        f"Rows selected for brand '{args.brand}': "
        f"{len(brand_dataframe)}"
    )

    if args.limit:
        brand_dataframe = brand_dataframe.head(args.limit)

    agent = SupportAgent(
        brand=args.brand,
        use_llm=args.use_llm,
    )

    predictions = []

    for message in brand_dataframe["message"]:
        result = agent.respond(str(message))
        predictions.append(result.to_dict())

    predictions_dataframe = pd.DataFrame(predictions)

    output_predictions = Path(
        settings.output_dir
    ) / "predictions.csv"

    predictions_dataframe.to_csv(
        output_predictions,
        index=False,
    )

    print(
        f"Saved predictions to {output_predictions}"
    )

    golden = load_golden_set(args.golden)

    metrics = {
        "brand": args.brand,
        "rows_processed": len(brand_dataframe),
        "intent_metrics": {},
        "escalation_metrics": {},
        "baselines": {},
    }

    if not golden.empty:
        print(
            f"Loaded {len(golden)} golden examples."
        )

        metrics["intent_metrics"] = evaluate_intents(
            golden,
            lambda text: classify_intent(text).intent,
        )

        metrics["escalation_metrics"] = evaluate_escalation(
            golden,
            lambda text: agent.respond(text).should_escalate,
        )

        metrics["baselines"] = {
            "majority": simple_majority_baseline(golden),
            "keyword": keyword_baseline(golden),
        }

        report = (
            "Intent classification report\n"
            "============================\n\n"
            + metrics["intent_metrics"].get(
                "classification_report",
                "No report",
            )
        )

        save_text(
            report,
            f"{settings.output_dir}/classification_report.txt",
        )

    save_json(
        metrics,
        f"{settings.output_dir}/metrics.json",
    )

    save_json(
        build_decision_log(agent),
        f"{settings.output_dir}/decision_log.json",
    )

    print(
        f"Saved metrics to {settings.output_dir}/metrics.json"
    )

    print("\nSample responses:\n")

    for prediction in predictions[:5]:
        print("-" * 80)
        print(f"Message: {prediction['message']}")
        print(f"Intent: {prediction['intent']}")
        print(
            f"Confidence: {prediction['confidence']}"
        )
        print(
            f"Escalate: {prediction['should_escalate']}"
        )
        print(
            f"Reason: {prediction['escalation_reason']}"
        )
        print(f"Reply: {prediction['response']}")


if __name__ == "__main__":
    main()
