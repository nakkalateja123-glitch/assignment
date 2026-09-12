from dataclasses import asdict, dataclass
from typing import Optional
import os

from .intents import IntentResult, classify_intent


@dataclass
class AgentResponse:
    message: str
    intent: str
    confidence: float
    should_escalate: bool
    escalation_reason: str
    response: str
    matched_keywords: list[str]
    method: str

    def to_dict(self) -> dict:
        return asdict(self)


class SupportAgent:
    """
    A transparent rule-based support agent.

    The system is intentionally conservative:
    - low-confidence requests are escalated
    - sensitive financial/account issues are escalated
    - abusive or high-risk requests are escalated
    - routine delivery questions can be auto-handled
    """

    def __init__(
        self,
        brand: str,
        use_llm: bool = False,
    ):
        self.brand = brand
        self.use_llm = use_llm

        self.response_templates = {
            "delivery_delay": (
                "Sorry your order is taking longer than expected. "
                "Please send us your order number by private message so "
                "we can check the latest tracking information."
            ),
            "refund_request": (
                "We can help check the refund status. Please send your "
                "order number and the email address associated with the "
                "purchase by private message."
            ),
            "payment_problem": (
                "We are sorry about the payment issue. Please do not share "
                "card details publicly. Send us a private message with your "
                "order reference so our billing team can investigate."
            ),
            "account_access": (
                "We can help you regain access. Please use the password "
                "reset option first. If that does not work, send us a "
                "private message so an agent can verify the account safely."
            ),
            "technical_issue": (
                "Sorry you are experiencing this issue. Please tell us the "
                "device, app or website version, and the exact error message "
                "by private message so we can investigate."
            ),
            "cancellation": (
                "We can check whether cancellation is still possible. "
                "Please send your order or subscription reference by private "
                "message."
            ),
            "product_question": (
                "Thanks for your question. Please share the specific product "
                "or service you are asking about, and we will provide the "
                "current details."
            ),
            "complaint": (
                "We are sorry that your experience has been disappointing. "
                "Please send us a private message with the relevant order "
                "details so a specialist can review this."
            ),
            "other": (
                "Thanks for contacting us. We need a little more information "
                "to understand the issue. Please send the relevant order or "
                "account details by private message."
            ),
        }

    def _risk_reason(
        self,
        message: str,
        result: IntentResult,
    ) -> Optional[str]:
        normalized = message.lower()

        high_risk_terms = [
            "fraud",
            "scam",
            "stolen",
            "identity theft",
            "legal action",
            "lawyer",
            "lawsuit",
            "police",
            "threat",
            "suicide",
            "self harm",
        ]

        for term in high_risk_terms:
            if term in normalized:
                return f"High-risk keyword detected: {term}"

        if result.confidence < 0.50:
            return "Low intent-classification confidence"

        if result.intent in {
            "payment_problem",
            "refund_request",
            "account_access",
        }:
            return (
                "Sensitive billing or account issue requires "
                "human verification"
            )

        if result.intent in {
            "complaint",
            "other",
        }:
            return "Issue is ambiguous or requires human judgment"

        return None

    def _llm_reply(
        self,
        message: str,
        intent: str,
    ) -> Optional[str]:
        if not self.use_llm:
            return None

        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            return None

        try:
            from openai import OpenAI

            client = OpenAI(api_key=api_key)

            prompt = f"""
You are a professional customer-support agent for {self.brand}.

Customer message:
{message}

Detected intent:
{intent}

Write a short, empathetic, safe public reply.
Do not request passwords, full card numbers, or sensitive personal data.
Ask the customer to use private message when order/account information
is needed. Do not invent policies, refunds, delivery dates, or tracking data.
"""

            response = client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                temperature=0.2,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You write concise customer-support replies."
                        ),
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
            )

            return response.choices[0].message.content.strip()

        except Exception:
            return None

    def respond(self, message: str) -> AgentResponse:
        result = classify_intent(message)

        escalation_reason = self._risk_reason(
            message,
            result,
        )

        should_escalate = escalation_reason is not None

        response = self._llm_reply(
            message,
            result.intent,
        )

        method = "llm"

        if not response:
            response = self.response_templates.get(
                result.intent,
                self.response_templates["other"],
            )
            method = "template"

        return AgentResponse(
            message=message,
            intent=result.intent,
            confidence=result.confidence,
            should_escalate=should_escalate,
            escalation_reason=escalation_reason or "",
            response=response,
            matched_keywords=result.matched_keywords,
            method=method,
        )
