#include "agents.hpp"

namespace jass {

bool parse_agent(const std::string& name, AgentKind& out) {
    if (name == "random") { out = AgentKind::Random; return true; }
    if (name == "heuristic") { out = AgentKind::Heuristic; return true; }
    return false;
}

const char* agent_name(AgentKind k) { return k == AgentKind::Random ? "random" : "heuristic"; }

int trump_score(Cards hand, int mode) {
    //                        A   K  Q  J  10  9  8  7  6
    static constexpr int kTrump[9] = {15, 10, 7, 25, 6, 19, 5, 5, 5};
    static constexpr int kObeW[9] = {14, 10, 8, 7, 5, 0, 5, 0, 0};
    static constexpr int kUneW[9] = {0, 2, 1, 1, 5, 5, 7, 9, 11};
    int s = 0;
    for (Cards h = hand; h; h &= h - 1) {
        int c = lowest(h);
        if (mode == kObe) s += kObeW[rank_of(c)];
        else if (mode == kUne) s += kUneW[rank_of(c)];
        else if (suit_of(c) == mode) s += kTrump[rank_of(c)];
    }
    return s;
}

namespace {

constexpr int kPushThreshold = 68;

int strength(int card, int trump, int lead) {
    // Comparable strength of `card` in a trick led with suit `lead`; -1 if it cannot win.
    if (trump < kObe && suit_of(card) == trump) return 100 + kTrumpStrength[rank_of(card)];
    if (suit_of(card) != lead) return -1;
    return trump == kUne ? rank_of(card) : 8 - rank_of(card);
}

int value(int card, int trump) { return kPoints.v[trump][card]; }

// Card of `set` minimising (or maximising) point value, ties broken by weaker card.
int pick_by_value(Cards set, int trump, bool maximise) {
    int best = -1, best_key = 0;
    for (Cards s = set; s; s &= s - 1) {
        int c = lowest(s);
        int key = value(c, trump) * 16 + (trump < kObe && suit_of(c) == trump ? 8 : 0);
        if (!maximise) key = -key;
        if (best < 0 || key > best_key) { best = c; best_key = key; }
    }
    return best;
}

// Is `card` the strongest still-unplayed card of its suit (from this player's view)?
bool is_master(int card, const Round& r) {
    const int s = suit_of(card);
    const Cards remaining = kSuitMask[s] & ~r.played;
    const int mode_lead = s;
    for (Cards t = remaining; t; t &= t - 1) {
        int o = lowest(t);
        if (o != card && strength(o, r.trump, mode_lead) > strength(card, r.trump, mode_lead)) return false;
    }
    return true;
}

int heuristic_card(const Round& r, Rng& rng) {
    const Cards legal = r.legal();
    const int trump = r.trump;
    if (popcount(legal) == 1) return lowest(legal);

    if (r.n_in_trick == 0) {
        // Lead a master card (non-trump first, keeps trumps for later); else a cheap card.
        for (int pass = 0; pass < 2; ++pass)
            for (Cards s = legal; s; s &= s - 1) {
                int c = lowest(s);
                bool is_trump = trump < kObe && suit_of(c) == trump;
                if (is_trump == (pass == 1) && is_master(c, r)) return c;
            }
        return pick_by_value(legal, trump, false);
    }

    const int lead = suit_of(r.trick[0]);
    int win_pos = 0;
    for (int i = 1; i < r.n_in_trick; ++i)
        if (strength(r.trick[i], trump, lead) > strength(r.trick[win_pos], trump, lead)) win_pos = i;
    const bool partner_winning = (r.n_in_trick - win_pos) == 2;
    const int best_strength = strength(r.trick[win_pos], trump, lead);
    int trick_pts = 0;
    for (int i = 0; i < r.n_in_trick; ++i) trick_pts += value(r.trick[i], trump);

    const bool partner_safe = partner_winning &&
        (r.n_in_trick == 3 || is_master(r.trick[win_pos], r) || best_strength >= 100);
    if (partner_safe) {
        // Schmieren: give points to the partner, but don't waste trumps.
        Cards non_trump = trump < kObe ? legal & ~kSuitMask[trump] : legal;
        return pick_by_value(non_trump ? non_trump : legal, trump, !!non_trump);
    }

    // Cheapest card that takes the lead; only trump in if it is worth it.
    int best = -1, best_key = 1 << 30;
    for (Cards s = legal; s; s &= s - 1) {
        int c = lowest(s);
        int st = strength(c, trump, lead);
        if (st <= best_strength) continue;
        if (st >= 100 && best_strength < 100 && trick_pts < 10 && r.n_in_trick < 3) continue;
        int key = st + value(c, trump) * 4;
        if (key < best_key) { best = c; best_key = key; }
    }
    if (best >= 0) return best;
    (void)rng;
    return pick_by_value(legal, trump, false);
}

}  // namespace

int choose_trump(AgentKind k, const Round& r, Rng& rng) {
    const Cards hand = r.hands[r.trump_chooser()];
    if (k == AgentKind::Random) {
        int n = r.pushed ? 6 : 7;
        int t = static_cast<int>(rng.below(n));
        return t == 6 ? kPush : t;
    }
    int best = 0, best_score = -1;
    for (int m = 0; m < 6; ++m) {
        int s = trump_score(hand, m);
        if (s > best_score) { best = m; best_score = s; }
    }
    if (!r.pushed && best_score < kPushThreshold) return kPush;
    return best;
}

int choose_card(AgentKind k, const Round& r, Rng& rng) {
    if (k == AgentKind::Random) return rng.pick(r.legal());
    return heuristic_card(r, rng);
}

}  // namespace jass
