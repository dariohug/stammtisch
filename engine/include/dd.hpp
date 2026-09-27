// Double-dummy (perfect-information) solver: exact minimax value of a position
// when all four hands are known. Building block for PIMC search and game analysis.
#pragma once
#include <cstdint>
#include <vector>

#include "jass.hpp"

namespace jass {

struct Position {
    Cards hands[4];
    int trick[4];
    int n_in_trick;
    int leader;       // seat that led the current trick
    int trump;
    int tricks_done;  // completed tricks
    int won_mask;     // bit t: team t has won at least one trick so far

    int to_play() const {
        int s = leader;
        for (int i = 0; i < n_in_trick; ++i) s = next_seat(s);
        return s;
    }
    // Current state of a round; hands taken from r.hands (so fill in a full deal first).
    static Position from_round(const Round& r);
};

class DDSolver {
public:
    explicit DDSolver(int tt_bits = 21);

    // Raw point difference (team 0 minus team 1) of the remaining play: current trick,
    // later tricks, last-trick bonus and match bonus.
    int value(const Position& p);
    // Exact value of every legal move for the player to act, from that player's team's
    // perspective. Returns the number of moves written.
    int move_values(const Position& p, int* cards, int* values);

    uint64_t nodes = 0;
    uint64_t iterations = 0;  // null-window searches

private:
    struct Entry {
        uint64_t key;
        int16_t lo, hi;
        int8_t best;
        int8_t depth;  // tricks remaining; deeper entries are kept preferentially
    };
    std::vector<Entry> tt_;  // buckets of two: [depth-preferred, always-replace]
    uint64_t mask_;
    int32_t history_[4][36][36]{};  // [n_in_trick][first card of trick or 35][card] cutoff counts

    int search(Position& p, uint64_t key, int alpha, int beta);
    int mtdf(Position& p, uint64_t key, int guess);
    int greedy_value(Position p) const;  // quick playout with the move-ordering policy (MTD(f) guess)
    Entry* probe(uint64_t full);
    void store(uint64_t full, int depth, int best, int alpha0, int beta0, int best_move);
    int order_moves(const Position& p, int seat, Cards legal, int tt_best, int* out) const;
};

// Strength position of a card within its suit under a trump mode (0 = strongest).
int suit_order(int card, int trump);

}  // namespace jass
