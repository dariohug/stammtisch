"""Smoke tests for the interactive tools (advise, play) — scripted, no terminal needed."""
import numpy as np
import pytest

from stammtisch import engine, record


def test_advise_card_and_trump(capsys):
    from stammtisch import advise
    hand = "RoA RoK RoU Ro9 Ro6 SchiA Schi10 Ei7 Sche6"
    advise.advise(hand, dealer="left", samples=4)
    out = capsys.readouterr().out
    assert "push" in out and "Obenabe" in out
    advise.advise(hand, dealer="left", trump="Rosen", played="", samples=4)
    out = capsys.readouterr().out
    assert "best" in out and "RoA" in out


def test_advise_rejects_wrong_turn():
    from stammtisch import advise
    with pytest.raises(SystemExit):
        # dealer = me -> forehand is the right-hand opponent, so it is not my turn to lead
        advise.advise("RoA RoK RoU Ro9 Ro6 SchiA Schi10 Ei7 Sche6", dealer="me", trump="Rosen", played="", samples=4)


def test_play_scripted_round(monkeypatch, tmp_path):
    from stammtisch import play
    envs = []
    real_init = engine.VecEnv.__init__

    def init(self, *a, **k):
        real_init(self, *a, **k)
        envs.append(self)

    def fake_ask(question, choices=None, show_choices=True):
        if choices:
            return "schieben" if "schieben" in choices else "obenabe"
        legal = np.flatnonzero(envs[-1].legal_mask()[0][:36])
        return engine.card_name(int(legal[0]))

    monkeypatch.setattr(engine.VecEnv, "__init__", init)
    monkeypatch.setattr(play.Prompt, "ask", staticmethod(fake_ask))
    monkeypatch.setattr(play, "ROOT", tmp_path)
    play.play(bots="heuristic", samples=4, seed=9)
    saved = list((tmp_path / "runs" / "games").glob("*.jass"))
    assert len(saved) == 1
    g = record.parse(saved[0].read_text())
    assert len(g.cards) == 36 and len(set(g.cards)) == 36
