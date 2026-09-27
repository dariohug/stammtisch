"""Recorded games: a small text format that is quick to type after (or during) a game.

    # Stammtisch, 27.9.2026, Runde 3
    players: Dario Anna Beat Chris    # in playing order (counter-clockwise); partners sit opposite
    me: Dario                          # whose decisions to review (default: first player)
    dealer: Chris
    trump: Rosen                       # Eichel/Ecken, Rosen/Herz, Schilten/Schaufel, Schellen/Kreuz, Obenabe, Undeufe
    pushed: no                         # yes = forehand pushed, partner declared
    rules: swisslos                    # swisslos (Weis, Stöck, x1/x2/x3) | plain
    hands:                             # optional when all 9 tricks are listed
      Dario: RoA RoK Sche9 ...
    tricks:                            # cards in the order they were played, leader first
      Anna: EiA  Beat: Ei6  Chris: EiK  Dario: RoU
      Dario: RoA Anna: Ro6 Beat: Ro7 Chris: Ro9      # "name:" labels are optional but checked

Cards: suit + rank. Suits: Eichel/Ei, Rosen/Ro, Schilten/Schi, Schellen/Sche (Swisslos mapping to
♦ ♥ ♠ ♣), or D/H/S/C, ♦♥♠♣, Ecken, Herz, Schaufel, Kreuz, Karo, Pik. Ranks: A/Ass, K/König,
O/Ober/Q, U/Under/J/Bauer, B/Banner/10, 9 (Nell), 8, 7, 6.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .engine import card_index, card_name

SUIT_WORDS = [  # (prefix, suit index) — longest prefixes first where they overlap
    ("schellen", 3), ("sche", 3), ("schilten", 2), ("schilte", 2), ("schi", 2), ("schaufel", 2), ("schau", 2),
    ("eichel", 0), ("eicheln", 0), ("ei", 0), ("ecken", 0), ("ecke", 0), ("karo", 0),
    ("rosen", 1), ("rose", 1), ("ro", 1), ("herz", 1), ("kreuz", 3), ("kr", 3), ("pik", 2),
    ("♦", 0), ("♥", 1), ("♠", 2), ("♣", 3), ("d", 0), ("h", 1), ("s", 2), ("c", 3),
]
RANK_WORDS = {
    "a": "A", "ass": "A", "as": "A", "k": "K", "könig": "K", "koenig": "K", "konig": "K",
    "q": "Q", "o": "Q", "ober": "Q", "dame": "Q", "j": "J", "u": "J", "under": "J", "bauer": "J", "puur": "J",
    "10": "10", "b": "10", "banner": "10", "zehn": "10", "9": "9", "nell": "9", "8": "8", "7": "7", "6": "6",
}
TRUMP_WORDS = {
    "eichel": 0, "eicheln": 0, "ecken": 0, "ecke": 0, "karo": 0, "d": 0, "♦": 0,
    "rosen": 1, "rose": 1, "herz": 1, "h": 1, "♥": 1,
    "schilten": 2, "schilte": 2, "schaufel": 2, "pik": 2, "s": 2, "♠": 2,
    "schellen": 3, "schelle": 3, "kreuz": 3, "c": 3, "♣": 3,
    "obenabe": 4, "obeabe": 4, "obe": 4, "undeufe": 5, "undenufe": 5, "une": 5, "uneufe": 5,
}
GERMAN_SUIT = ["Ei", "Ro", "Schi", "Sche"]
GERMAN_RANK = {"A": "A", "K": "K", "Q": "O", "J": "U", "10": "B", "9": "9", "8": "8", "7": "7", "6": "6"}
SUIT_SYMBOL = "♦♥♠♣"


def parse_card(token: str) -> int:
    t = token.strip().lower().replace("-", "").replace("_", "")
    for prefix, suit in sorted(SUIT_WORDS, key=lambda x: -len(x[0])):
        if t.startswith(prefix):
            rank = RANK_WORDS.get(t[len(prefix):])
            if rank:
                return card_index("DHSC"[suit] + rank)
    raise ValueError(f"unknown card {token!r} (e.g. RoA, SchiU, Ei9, H10, ♠K)")


def pretty(card: int, style: str = "german") -> str:
    name = card_name(card)
    if style == "german":
        return GERMAN_SUIT[card // 9] + GERMAN_RANK[name[1:]]
    return SUIT_SYMBOL[card // 9] + name[1:]


@dataclass
class Game:
    players: list[str]                      # in playing order
    dealer: int                             # index into players
    trump: int
    pushed: bool
    cards: list[int]                        # play order
    me: int = 0                             # index into players
    rules: str = "swisslos"
    hands: dict[int, list[int]] = field(default_factory=dict)  # player index -> cards (optional)
    labels: list[int | None] = field(default_factory=list)     # stated player per card (optional)
    title: str = ""

    # Engine seats: players[i] sits at seat (-i) mod 4, so that next_seat = (p + 3) % 4
    # follows the playing order and partners (i, i+2) share a team.
    @staticmethod
    def seat(i: int) -> int:
        return (-i) % 4

    @staticmethod
    def player(seat: int) -> int:
        return (-seat) % 4


def _name_index(players: list[str], name: str) -> int:
    for i, p in enumerate(players):
        if p.lower() == name.lower():
            return i
    raise ValueError(f"unknown player {name!r}; players are {players}")


def parse(text: str) -> Game:
    fields: dict[str, str] = {}
    hands_raw: dict[str, str] = {}
    tricks: list[str] = []
    section = None
    title = ""
    for raw in text.splitlines():
        if raw.strip().startswith("#") and not title and not fields:
            title = raw.strip().lstrip("#").strip()
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        m = re.match(r"^(\w+)\s*:\s*(.*)$", line.strip())
        indented = raw[:1] in (" ", "\t")
        if m and not indented and m.group(1).lower() in {"players", "me", "dealer", "trump", "pushed", "rules",
                                                         "hands", "tricks"}:
            key, val = m.group(1).lower(), m.group(2).strip()
            section = key if key in ("hands", "tricks") else None
            if section is None:
                fields[key] = val
            elif section == "tricks" and val:
                tricks.append(val)
            continue
        if section == "hands":
            name, _, cards = line.strip().partition(":")
            hands_raw[name.strip()] = cards
        elif section == "tricks":
            tricks.append(re.sub(r"^\s*\d+\s*[:.)]\s*", "", line.strip()))
        else:
            raise ValueError(f"cannot parse line: {raw!r}")

    for key in ("players", "dealer", "trump"):
        if key not in fields:
            raise ValueError(f"missing '{key}:'")
    players = re.split(r"[,\s]+", fields["players"].strip())
    if len(players) != 4:
        raise ValueError("players: needs exactly four names in playing order")
    trump_word = fields["trump"].strip().lower()
    if trump_word not in TRUMP_WORDS:
        raise ValueError(f"unknown trump {fields['trump']!r}")
    cards: list[int] = []
    labels: list[int | None] = []
    for t in tricks:
        for tok in re.findall(r"(?:([^\s:]+)\s*:\s*)?([^\s:]+)", t):
            name, card = tok
            labels.append(_name_index(players, name) if name else None)
            cards.append(parse_card(card))
    hands = {_name_index(players, n): [parse_card(c) for c in v.replace(",", " ").split()] for n, v in hands_raw.items()}
    return Game(players=players, dealer=_name_index(players, fields["dealer"]), trump=TRUMP_WORDS[trump_word],
                pushed=fields.get("pushed", "no").strip().lower() in ("yes", "ja", "true", "1", "y"),
                cards=cards, me=_name_index(players, fields.get("me", players[0])),
                rules=fields.get("rules", "swisslos").strip().lower(), hands=hands, labels=labels, title=title)


def from_swisslos(d: dict, g: int, me_seat: int = 0) -> Game:
    """A logged Swisslos game (index g of data.load()) as a Game, seat names P0..P3."""
    order = [(me_seat - i) % 4 for i in range(4)]  # playing order starting with me_seat
    players = [f"Seat{s}" for s in order]
    return Game(players=players, dealer=order.index(int(d["dealer"][g])), trump=int(d["trump"][g]),
                pushed=bool(d["forehand"][g] == 0), cards=[int(c) for c in d["cards"][g]], me=0, rules="plain",
                title=f"Swisslos game #{g}")


def dump(game: Game) -> str:
    """Text format of a game (inverse of parse)."""
    trump_name = ["Eichel", "Rosen", "Schilten", "Schellen", "Obenabe", "Undeufe"][game.trump]
    out = [f"# {game.title}"] if game.title else []
    out += [f"players: {' '.join(game.players)}", f"me: {game.players[game.me]}",
            f"dealer: {game.players[game.dealer]}", f"trump: {trump_name}",
            f"pushed: {'yes' if game.pushed else 'no'}", f"rules: {game.rules}", "tricks:"]
    for t in range(0, len(game.cards), 4):
        out.append("  " + "  ".join(pretty(c) for c in game.cards[t:t + 4]))
    return "\n".join(out) + "\n"


def self_play(agent: str = "heuristic", seed: int = 0, rules: str = "swisslos",
              players: tuple[str, ...] = ("Dario", "Anna", "Beat", "Chris")) -> Game:
    """A full round played by an engine agent, as a Game (for examples and tests)."""
    from .engine import VecEnv, mask_to_cards
    env = VecEnv(1, seed=seed, rules=rules)
    while int(env.info[0, 4]) < 35:
        env.step(env.builtin_actions(agent), auto_reset=False)
    dealer, trump, pushed = int(env.info[0, 3]), int(env.info[0, 0]), bool(env.info[0, 2])
    last = mask_to_cards(int(env.hands[0, env.seat[0]]))[0]
    cards = [int(c) for c in env.history[0, :35]] + [last]
    return Game(players=list(players), dealer=Game.player(dealer), trump=trump, pushed=pushed,
                cards=cards, rules=rules, title=f"self-play ({agent}, seed {seed})")
