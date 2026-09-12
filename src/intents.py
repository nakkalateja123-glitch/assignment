from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class IntentResult:
    intent: str
    confidence: float
    matched_keywords: List[str]


INTENT_DEFINITIONS: Dict[str, Dict[str, object]] = {
    "delivery_delay": {
        "description": "The customer says an order has not arrived or is delayed.",
        "keywords": [
            "late",
            "delayed",
            "delay",
            "not arrived",
            "where is my order",
            "still waiting",
            "delivery",
            "shipping",
            "shipment",
            "tracking",
        ],
    },
    "refund_request": {
        "description": "The customer requests a refund or asks about refund status.",
        "keywords": [
            "refund",
            "money back",
            "reimburse",
            "reimbursement",
            "charged back",
            "return my money",
        ],
    },
    "payment_problem": {
        "description": "The customer reports a payment, card, billing, or charge issue.",
        "keywords": [
            "payment",
            "card declined",
            "credit card",
            "debit card",
            "charged",
            "billing",
            "invoice",
            "double charge",
            "wrong charge",
        ],
    },
    "account_access": {
        "description": "The customer cannot sign in or access their account.",
        "keywords": [
            "login",
            "log in",
            "sign in",
            "signin",
            "password",
            "locked out",
            "account access",
            "cannot access",
            "can't access",
        ],
    },
    "technical_issue": {
        "description": "The customer reports that a product, service, website, or app does not work.",
        "keywords": [
            "not working",
            "doesn't work",
            "does not work",
            "broken",
            "error",
            "bug",
            "crash",
            "crashed",
            "app",
            "website",
            "feature",
        ],
    },
    "cancellation": {
        "description": "The customer wants to cancel an order, subscription, or service.",
        "keywords": [
            "cancel",
            "cancellation",
            "stop my order",
            "end subscription",
            "unsubscribe",
        ],
    },
    "product_question": {
        "description": "The customer asks a general question about a product or service.",
        "keywords": [
            "how much",
            "price",
            "available",
            "availability",
            "do you offer",
            "what is",
            "information",
            "question",
        ],
    },
    "complaint": {
        "description": "The customer expresses dissatisfaction without a clearly classifiable request.",
        "keywords": [
            "terrible",
            "awful",
            "unacceptable",
            "disappointed",
            "angry",
            "worst",
            "poor service",
            "complaint",
        ],
    },
    "other": {
        "description": "The message does not fit another supported intent.",
        "keywords": [],
    },
}


def normalize_text(text: str) -> str:
    return " ".join(str(text).lower().strip().split())


def classify_intent(text: str) -> IntentResult:
    normalized = normalize_text(text)

    scores: Dict[str, int] = {}

    for intent_name, definition in INTENT_DEFINITIONS.items():
        keywords = definition["keywords"]
        score = 0

        for keyword in keywords:
            if keyword in normalized:
                score += 1

        scores[intent_name] = score

    best_intent = max(scores, key=scores.get)
    best_score = scores[best_intent]

    if best_score == 0:
        return IntentResult(
            intent="other",
            confidence=0.20,
            matched_keywords=[],
        )

    all_keywords = INTENT_DEFINITIONS[best_intent]["keywords"]
    matched = [
        keyword
        for keyword in all_keywords
        if keyword in normalized
    ]

    confidence = min(0.95, 0.50 + (best_score * 0.12))

    return IntentResult(
        intent=best_intent,
        confidence=round(confidence, 3),
        matched_keywords=matched,
    )


def get_intent_names() -> List[str]:
    return list(INTENT_DEFINITIONS.keys())


def get_intent_descriptions() -> Dict[str, str]:
    return {
        key: str(value["description"])
        for key, value in INTENT_DEFINITIONS.items()
    }
