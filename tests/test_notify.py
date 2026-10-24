import pytest

from quantlab.execution import NotifyingBroker, Order, OrderStatus, PaperBroker
from quantlab.notify import DiscordNotifier, NullNotifier, notifier_from_env

URL = "https://discord.com/api/webhooks/123/secrettoken"


class Rec:
    """Records the calls of a notifier."""
    def __init__(self):
        self.calls = []

    def send(self, title, body="", *, fields=None, level="info"):
        self.calls.append((title, body, dict(fields or []), level))
        return True


def test_webhook_url_is_validated_and_never_printed():
    for bad in ("http://discord.com/api/webhooks/1/x", "https://evil.example/api/webhooks/1/x",
                "https://discord.com/other/1/x", "not a url"):
        with pytest.raises(ValueError):
            DiscordNotifier(bad)
    n = DiscordNotifier(URL)
    assert "secrettoken" not in repr(n) and "secrettoken" not in str(n)


def test_payload_disables_mentions_and_truncates():
    seen = {}
    n = DiscordNotifier(URL, transport=lambda u, p: (seen.update(url=u, payload=p) or (204, {})))
    assert n.send("T" * 500, "B" * 9000, fields=[("a", "x" * 5000)] * 40, level="error") is True
    p = seen["payload"]
    assert p["allowed_mentions"] == {"parse": []}                       # untrusted text cannot ping anyone
    e = p["embeds"][0]
    assert len(e["title"]) == 256 and len(e["description"]) == 4000
    assert len(e["fields"]) == 25 and len(e["fields"][0]["value"]) == 1024


def test_rate_limit_is_retried_with_retry_after_then_gives_up():
    calls, slept = [], []
    def transport(u, p):
        calls.append(1)
        return (429, {"Retry-After": "2"}) if len(calls) < 3 else (204, {})
    n = DiscordNotifier(URL, transport=transport, sleep=slept.append, max_retries=2)
    assert n.send("t") is True and slept == [2.0, 2.0]
    always = DiscordNotifier(URL, transport=lambda u, p: (429, {"retry-after": "1"}), sleep=lambda s: None)
    assert always.send("t") is False


def test_notification_failure_never_raises():
    def boom(u, p):
        raise OSError("network down")
    assert DiscordNotifier(URL, transport=boom).send("t") is False
    assert DiscordNotifier(URL, transport=lambda u, p: (500, {})).send("t") is False


def test_env_factory(monkeypatch):
    monkeypatch.delenv("DISCORD_WEBHOOK_URL", raising=False)
    assert isinstance(notifier_from_env(), NullNotifier)
    monkeypatch.setenv("DISCORD_WEBHOOK_URL", URL)
    assert isinstance(notifier_from_env(), DiscordNotifier)
    assert isinstance(notifier_from_env(enabled=False), NullNotifier)
    assert NullNotifier().send("x") is False


def test_notifying_broker_announces_each_status_once_and_labels_the_mode():
    rec = Rec()
    b = NotifyingBroker(PaperBroker(1000), rec, "PAPER")
    o = Order("id1", "SPY", "BUY", 2)
    b.submit_order(o)
    b.submit_order(o)                                                   # duplicate: no second message
    assert len(rec.calls) == 1 and rec.calls[0][0].startswith("[PAPER] Ordre soumis")
    assert rec.calls[0][2]["Symbole"] == "SPY" and "SIMULATION" in rec.calls[0][1]
    b.inner.process_fills({"SPY": 100.0})
    assert b.poll() == 1 and b.poll() == 0                              # fill announced once
    assert rec.calls[1][0].startswith("[PAPER] Ordre exécuté") and rec.calls[1][3] == "success"
    assert float(rec.calls[1][2]["Prix moyen"]) > 100


def test_rejection_and_cancel_are_announced():
    rec = Rec()
    b = NotifyingBroker(PaperBroker(50), rec, "PAPER")
    b.submit_order(Order("big", "A", "BUY", 10))
    b.submit_order(Order("nopx", "B", "BUY", 1))
    b.inner.process_fills({"A": 100.0})
    b.poll()
    assert b.inner.get_order("big").status == OrderStatus.REJECTED
    assert any("rejeté" in c[0] and c[2]["Raison"] == "insufficient cash" and c[3] == "error" for c in rec.calls)
    b.cancel_order("nopx")
    assert any("annulé" in c[0] for c in rec.calls)


def test_live_mode_is_visibly_different():
    rec = Rec()
    NotifyingBroker(PaperBroker(1000), rec, "LIVE").submit_order(Order("i", "A", "BUY", 1))
    assert "ORDRE RÉEL" in rec.calls[0][1] and rec.calls[0][0].startswith("[LIVE]")


def test_restored_broker_does_not_reannounce_history():
    rec = Rec()
    b1 = NotifyingBroker(PaperBroker(1000), Rec(), "PAPER")
    b1.submit_order(Order("old", "A", "BUY", 1))
    restored = PaperBroker.from_dict(b1.inner.to_dict())
    nb = NotifyingBroker(restored, rec, "PAPER")
    assert nb.poll() == 0 and rec.calls == []
    nb.submit_order(Order("old", "A", "BUY", 1))                         # replay of the same decision
    assert rec.calls == []
