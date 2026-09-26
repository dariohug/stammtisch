// stammtisch — fast Schieber Jass engine.
//
// Encoding is identical to HSLU jass-kit so that data and agents are interchangeable:
//   card  = suit * 9 + rank,  suit: 0=D(Schellen) 1=H(Rosen) 2=S(Schilten) 3=C(Eichel)
//                             rank: 0=A 1=K 2=Q 3=J 4=10 5=9 6=8 7=7 8=6
//   trump = 0..3 suit, 4=Obenabe, 5=Undeufe, 10=push (schieben)
//   seats play counter-clockwise: next(p) = (p + 3) % 4, teams {0,2} and {1,3}.
// Hands are 36-bit sets in a uint64_t.
#pragma once
#include <array>
#include <cstdint>

namespace jass {

using Cards = uint64_t;

constexpr int kNumCards = 36;
constexpr int kObe = 4, kUne = 5, kPush = 10;
constexpr int kJ = 3, kNine = 5;
constexpr Cards kAll = (Cards{1} << 36) - 1;
constexpr Cards kSuitMask[4] = {0x1FFull, 0x1FFull << 9, 0x1FFull << 18, 0x1FFull << 27};

constexpr int suit_of(int c) { return c / 9; }
constexpr int rank_of(int c) { return c % 9; }
constexpr Cards bit(int c) { return Cards{1} << c; }
constexpr int next_seat(int p) { return (p + 3) & 3; }
constexpr int partner(int p) { return (p + 2) & 3; }
constexpr int team_of(int p) { return p & 1; }

inline int popcount(Cards c) { return __builtin_popcountll(c); }
inline int lowest(Cards c) { return __builtin_ctzll(c); }

// Strength of a trump card, higher wins: J > 9 > A > K > Q > 10 > 8 > 7 > 6.
constexpr int kTrumpStrength[9] = {6, 5, 4, 8, 3, 7, 2, 1, 0};

// Card points per trump mode (index 0..3 = trump suit, 4 = Obenabe, 5 = Undeufe).
struct PointTable {
    std::array<std::array<int8_t, kNumCards>, 6> v{};
    constexpr PointTable() {
        constexpr int8_t plain[9] = {11, 4, 3, 2, 10, 0, 0, 0, 0};
        constexpr int8_t trump[9] = {11, 4, 3, 20, 10, 14, 0, 0, 0};
        constexpr int8_t obe[9] = {11, 4, 3, 2, 10, 0, 8, 0, 0};
        constexpr int8_t une[9] = {0, 4, 3, 2, 10, 0, 8, 0, 11};
        for (int t = 0; t < 6; ++t)
            for (int c = 0; c < kNumCards; ++c) {
                int s = c / 9, r = c % 9;
                v[t][c] = t == 4 ? obe[r] : t == 5 ? une[r] : (s == t ? trump[r] : plain[r]);
            }
    }
};
inline constexpr PointTable kPoints{};

constexpr int kLastTrickBonus = 5;
constexpr int kMatchBonus = 100;  // all nine tricks: 157 + 100 = 257

// Legal cards for `hand` given the cards already in the current trick.
// Official Schieber rules: follow suit; you may always trump instead (except
// under-trumping); the trump Jack (Puur) never has to be played.
Cards legal_cards(Cards hand, const int* trick, int n_in_trick, int trump);

// Index (0..3, position in trick) of the winning card of a full trick.
int trick_winner_pos(const int* trick, int trump);

inline int trick_points(const int* trick, int trump, bool last) {
    const auto& pv = kPoints.v[trump];
    return pv[trick[0]] + pv[trick[1]] + pv[trick[2]] + pv[trick[3]] + (last ? kLastTrickBonus : 0);
}

// Full state of one Schieber round (one deal, 9 tricks).
struct Round {
    Cards hands[4]{};
    Cards played = 0;
    int dealer = 0;
    int trump = -1;
    bool pushed = false;
    int declarer = -1;
    int trick[4]{-1, -1, -1, -1};
    int n_in_trick = 0;
    int trick_first = 0;
    int n_tricks = 0;  // completed tricks
    int to_play = 0;
    int points[2]{0, 0};
    int tricks_won[2]{0, 0};
    int history[36];  // cards in play order
    int winners[9];

    void reset(const Cards h[4], int dealer_seat);
    int forehand() const { return next_seat(dealer); }
    // Trump phase: seat to decide, or -1 if trump is set.
    int trump_chooser() const { return trump >= 0 ? -1 : (pushed ? partner(forehand()) : forehand()); }
    void choose_trump(int t);  // t in 0..5, or kPush (only forehand, once)
    Cards legal() const { return legal_cards(hands[to_play], trick, n_in_trick, trump); }
    void play(int card);       // caller guarantees legality
    bool done() const { return n_tricks == 9; }
    // Final score incl. match bonus (before trump multiplier).
    int score(int team) const {
        return points[team] + (tricks_won[team] == 9 ? kMatchBonus : 0);
    }
};

// Splitmix/xoshiro-style RNG, fast and reproducible.
struct Rng {
    uint64_t s[4];
    explicit Rng(uint64_t seed);
    uint64_t next();
    uint32_t below(uint32_t n) { return static_cast<uint32_t>((next() >> 32) * n >> 32); }
    // Uniform random element of a non-empty set.
    int pick(Cards set);
};

void deal(Rng& rng, Cards out[4]);

}  // namespace jass
