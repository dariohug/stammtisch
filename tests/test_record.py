import pytest

from stammtisch import record
from stammtisch.engine import card_index as ci

@pytest.mark.parametrize("token,card", [
    ("RoA", "HA"), ("rosenass", "HA"), ("H10", "H10"), ("♥10", "H10"), ("RoB", "H10"),
    ("SchiU", "SJ"), ("schilten-under", "SJ"), ("Sche9", "C9"), ("schellenober", "CQ"),
    ("Ei6", "D6"), ("eichelkönig", "DK"), ("KrA", "CA"), ("pikdame", "SQ"), ("herzbauer", "HJ"),
])
def test_parse_card(token, card):
    assert record.parse_card(token) == ci(card)


def test_parse_card_rejects_garbage():
    with pytest.raises(ValueError):
        record.parse_card("RoZ")


def test_example_file_parses_and_round_trips():
    from pathlib import Path
    text = (Path(__file__).parent.parent / "examples" / "example_game.jass").read_text()
    g = record.parse(text)
    assert len(g.cards) == 36 and g.players == ["Dario", "Anna", "Beat", "Chris"]
    g2 = record.parse(record.dump(g))
    assert (g2.cards, g2.trump, g2.dealer, g2.pushed) == (g.cards, g.trump, g.dealer, g.pushed)


def test_wrong_player_label_is_reported():
    from stammtisch import analyze
    g = record.self_play("heuristic", seed=2)
    g.labels = [None] * 36
    g.labels[5] = (analyze._seats_of_cards(g)[5] + 1) % 4
    with pytest.raises(ValueError, match="labelled"):
        analyze.analyze(g, samples=2)


def test_illegal_card_is_reported():
    """Swap two cards of the same player so that they fail to follow suit."""
    from stammtisch import analyze
    for seed in range(20):
        g = record.self_play("heuristic", seed=seed)
        owner = analyze._seats_of_cards(g)
        for k1 in range(36):
            if k1 % 4 == 0:
                continue
            lead = g.cards[k1 - k1 % 4] // 9
            x = g.cards[k1]
            if x // 9 != lead or lead == g.trump:
                continue
            for k2 in range((k1 // 4 + 1) * 4, 36):
                y = g.cards[k2]
                if owner[k2] == owner[k1] and y // 9 not in (lead, g.trump):
                    g.cards[k1], g.cards[k2] = y, x
                    with pytest.raises(ValueError, match="may not play"):
                        analyze.analyze(g, samples=2)
                    return
    pytest.fail("no suitable game found")


def test_review_separates_good_from_bad_play():
    """The analyzer must rank random play clearly worse than rule-based and learned play."""
    from stammtisch import analyze
    from stammtisch.engine import ROOT
    agents = ["random", "heuristic"] + (["net"] if (ROOT / "models" / "card_policy.bin").exists() else [])
    losses = {}
    for agent in agents:
        total = 0.0
        for seed in range(8):
            a = analyze.analyze(record.self_play(agent, seed=seed), samples=12, exact_left=5, who="all")
            total += sum(d.loss for d in a.decisions)
        losses[agent] = total / 8
    assert losses["random"] > 1.2 * losses["heuristic"], losses
    if "net" in losses:
        assert losses["heuristic"] > losses["net"], losses


def test_markdown_export():
    from stammtisch import analyze
    a = analyze.analyze(record.self_play("heuristic", seed=1), samples=4, exact_left=5)
    md = analyze.to_markdown(a)
    assert md.count("\n| ") >= 9 and "Review" in md
