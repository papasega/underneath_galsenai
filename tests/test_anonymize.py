from scripts.anonymize_chat import anonymize

CHAT = (
    "01/02/2026, 10:00 - Awa Diop: salut @Moussa voir https://x.org/a ou awa@mail.com, tel +221 77 123 45 67\n"
    "suite du message\n"
    "01/02/2026, 10:05 - TEL_0123abcd: merci Awa, et Fatou?\n"
)


def test_no_identifier_survives_and_threads_stay_linked() -> None:
    out = anonymize(CHAT, ["Fatou"], seed=1)
    for leaked in ("Awa", "Diop", "TEL_0123abcd", "Moussa", "x.org", "mail.com", "77 123", "Fatou", "10:00", "01/02"):
        assert leaked not in out
    first, cont, second = out.split("\n")[:3]
    assert first.startswith("2026-02 | U0") and cont == "suite du message" and "[mention]" in first
    assert second.split(":")[0] != first.split(":")[0]  # two different senders
    assert first.split(" | ")[1][:3] in second  # the quoted name maps to the same label as the sender


def test_labels_are_not_stable_across_runs() -> None:
    labels = {anonymize(CHAT, seed=s).split("|")[1].split(":")[0] for s in range(8)}
    assert len(labels) > 1
