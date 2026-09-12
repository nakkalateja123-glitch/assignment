from src.agent import SupportAgent


def test_delivery_message():
    agent = SupportAgent("AppleSupport")

    result = agent.respond(
        "My order is late and still has not arrived."
    )

    assert result.intent == "delivery_delay"
    assert result.response
    assert isinstance(result.should_escalate, bool)


def test_payment_message_escalates():
    agent = SupportAgent("AppleSupport")

    result = agent.respond(
        "I was charged twice for the same order."
    )

    assert result.intent == "payment_problem"
    assert result.should_escalate is True


def test_account_message_escalates():
    agent = SupportAgent("AppleSupport")

    result = agent.respond(
        "I cannot login to my account."
    )

    assert result.intent == "account_access"
    assert result.should_escalate is True


def test_unknown_message():
    agent = SupportAgent("AppleSupport")

    result = agent.respond(
        "Hello, I need some help."
    )

    assert result.intent in {
        "other",
        "complaint",
    }
    assert result.response
