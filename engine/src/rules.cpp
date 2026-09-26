#include "jass.hpp"

namespace jass {

namespace {

int trump_strength_of(int card) { return kTrumpStrength[rank_of(card)]; }

// For Obenabe / plain suits the natural order is A (rank 0) highest; for Undeufe 6 (rank 8).
bool beats_same_suit(int a, int b, int trump) {
    if (trump == kUne) return rank_of(a) > rank_of(b);
    return rank_of(a) < rank_of(b);
}

}  // namespace

Cards legal_cards(Cards hand, const int* trick, int n, int trump) {
    if (n == 0) return hand;
    const int lead = suit_of(trick[0]);
    const Cards lead_cards = hand & kSuitMask[lead];

    if (trump >= kObe) return lead_cards ? lead_cards : hand;

    const Cards trumps = hand & kSuitMask[trump];
    if (lead == trump) {
        // Must follow trump, unless the only trump held is the Puur.
        if (trumps == 0 || trumps == bit(trump * 9 + kJ)) return hand;
        return trumps;
    }

    // Strongest trump already in the trick (if any) forbids under-trumping.
    int best = -1;
    for (int i = 1; i < n; ++i)
        if (suit_of(trick[i]) == trump) best = best < 0 ? trick[i]
            : (trump_strength_of(trick[i]) > trump_strength_of(best) ? trick[i] : best);

    Cards higher_trumps = trumps;
    if (best >= 0) {
        higher_trumps = 0;
        for (Cards t = trumps; t; t &= t - 1) {
            int c = lowest(t);
            if (trump_strength_of(c) > trump_strength_of(best)) higher_trumps |= bit(c);
        }
    }
    if (lead_cards) return lead_cards | higher_trumps;
    // Cannot follow: anything except an under-trump — unless only trumps are left.
    if ((hand & ~trumps) == 0) return hand;
    return (hand & ~trumps) | higher_trumps;
}

int trick_winner_pos(const int* trick, int trump) {
    const int lead = suit_of(trick[0]);
    int win = 0;
    for (int i = 1; i < 4; ++i) {
        const int c = trick[i], w = trick[win];
        const bool c_trump = trump < kObe && suit_of(c) == trump;
        const bool w_trump = trump < kObe && suit_of(w) == trump;
        if (c_trump) {
            if (!w_trump || trump_strength_of(c) > trump_strength_of(w)) win = i;
        } else if (!w_trump && suit_of(c) == lead && beats_same_suit(c, w, trump)) {
            win = i;
        }
    }
    return win;
}

void Round::reset(const Cards h[4], int dealer_seat) {
    for (int p = 0; p < 4; ++p) hands[p] = h[p];
    played = 0;
    dealer = dealer_seat;
    trump = -1;
    pushed = false;
    declarer = -1;
    n_in_trick = 0;
    n_tricks = 0;
    trick_first = to_play = forehand();
    points[0] = points[1] = 0;
    tricks_won[0] = tricks_won[1] = 0;
}

void Round::choose_trump(int t) {
    if (t == kPush) {
        pushed = true;
        return;
    }
    declarer = trump_chooser();
    trump = t;
}

void Round::play(int card) {
    hands[to_play] &= ~bit(card);
    played |= bit(card);
    history[n_tricks * 4 + n_in_trick] = card;
    trick[n_in_trick++] = card;
    if (n_in_trick < 4) {
        to_play = next_seat(to_play);
        return;
    }
    const int pos = trick_winner_pos(trick, trump);
    int winner = trick_first;
    for (int i = 0; i < pos; ++i) winner = next_seat(winner);
    const bool last = n_tricks == 8;
    points[team_of(winner)] += trick_points(trick, trump, last);
    tricks_won[team_of(winner)]++;
    winners[n_tricks++] = winner;
    n_in_trick = 0;
    trick_first = to_play = winner;
}

Rng::Rng(uint64_t seed) {
    for (auto& x : s) {  // splitmix64 seeding
        seed += 0x9E3779B97F4A7C15ull;
        uint64_t z = seed;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        x = z ^ (z >> 31);
    }
}

uint64_t Rng::next() {  // xoshiro256**
    const uint64_t r = ((s[1] * 5) << 7 | (s[1] * 5) >> 57) * 9;
    const uint64_t t = s[1] << 17;
    s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3];
    s[2] ^= t;
    s[3] = (s[3] << 45) | (s[3] >> 19);
    return r;
}

int Rng::pick(Cards set) {
    int k = static_cast<int>(below(static_cast<uint32_t>(popcount(set))));
    while (k--) set &= set - 1;
    return lowest(set);
}

void deal(Rng& rng, Cards out[4]) {
    int deck[kNumCards];
    for (int i = 0; i < kNumCards; ++i) deck[i] = i;
    for (int i = kNumCards - 1; i > 0; --i) {
        int j = static_cast<int>(rng.below(static_cast<uint32_t>(i + 1)));
        int tmp = deck[i]; deck[i] = deck[j]; deck[j] = tmp;
    }
    for (int p = 0; p < 4; ++p) {
        out[p] = 0;
        for (int i = 0; i < 9; ++i) out[p] |= bit(deck[p * 9 + i]);
    }
}

}  // namespace jass
