#include "dd.hpp"

#include <algorithm>
#include <cstring>

namespace jass {

namespace {

constexpr int kInf = 10000;

// Rank indices (A K Q J 10 9 8 7 6 = 0..8) from strongest to weakest.
constexpr int kOrderTrump[9] = {3, 5, 0, 1, 2, 4, 6, 7, 8};
constexpr int kOrderUne[9] = {8, 7, 6, 5, 4, 3, 2, 1, 0};
constexpr int kOrderPlain[9] = {0, 1, 2, 3, 4, 5, 6, 7, 8};

const int* order_of(int suit, int trump) {
    if (trump < kObe && suit == trump) return kOrderTrump;
    return trump == kUne ? kOrderUne : kOrderPlain;
}

struct Tables {
    uint64_t card[4][36];
    uint64_t leader[4], won[4], trump[6], trick[4][36];
    int8_t ord[6][36];
    uint8_t suit_pts[6][4][512];  // card points of a 9-bit suit pattern
    Tables() {
        uint64_t s = 0x1234567890ABCDEFull;
        auto next = [&] {
            s += 0x9E3779B97F4A7C15ull;
            uint64_t z = s;
            z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
            z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
            return z ^ (z >> 31);
        };
        for (auto& p : card) for (auto& c : p) c = next();
        for (auto& x : leader) x = next();
        for (auto& x : won) x = next();
        for (auto& x : trump) x = next();
        for (auto& p : trick) for (auto& c : p) c = next();
        for (int t = 0; t < 6; ++t)
            for (int c = 0; c < 36; ++c) {
                const int* o = order_of(c / 9, t);
                for (int i = 0; i < 9; ++i)
                    if (o[i] == c % 9) ord[t][c] = static_cast<int8_t>(i);
            }
        for (int t = 0; t < 6; ++t)
            for (int su = 0; su < 4; ++su)
                for (int m = 0; m < 512; ++m) {
                    int sum = 0;
                    for (int r = 0; r < 9; ++r)
                        if (m >> r & 1) sum += kPoints.v[t][su * 9 + r];
                    suit_pts[t][su][m] = static_cast<uint8_t>(sum);
                }
    }
};
const Tables kT;

inline bool is_trump(int c, int trump) { return trump < kObe && suit_of(c) == trump; }

// Does card c beat the currently winning card w (lead suit `lead`)?
inline bool beats(int c, int w, int lead, int trump) {
    const bool ct = is_trump(c, trump), wt = is_trump(w, trump);
    if (ct != wt) return ct;
    if (ct) return kT.ord[trump][c] < kT.ord[trump][w];
    return suit_of(c) == lead && suit_of(w) == lead && kT.ord[trump][c] < kT.ord[trump][w];
}

inline int points_of(Cards c, int trump) {
    return kT.suit_pts[trump][0][c & 0x1FF] + kT.suit_pts[trump][1][(c >> 9) & 0x1FF] +
           kT.suit_pts[trump][2][(c >> 18) & 0x1FF] + kT.suit_pts[trump][3][(c >> 27) & 0x1FF];
}

inline uint64_t mix(uint64_t z) {
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

// Relative-rank key: per suit, the owners and point values of the remaining cards in
// strength order. Positions that differ only in which cards are already gone share it.
uint64_t abstract_key(const Position& p) {
    uint64_t h = mix(static_cast<uint64_t>(p.trump) * 64 + p.leader * 4 + p.won_mask + 0x51ED);
    const auto& pv = kPoints.v[p.trump];
    for (int s = 0; s < 4; ++s) {
        const int* o = order_of(s, p.trump);
        uint64_t sig = 1;
        for (int i = 0; i < 9; ++i) {
            const int c = s * 9 + o[i];
            const Cards b = bit(c);
            int owner;
            if (p.hands[0] & b) owner = 0;
            else if (p.hands[1] & b) owner = 1;
            else if (p.hands[2] & b) owner = 2;
            else if (p.hands[3] & b) owner = 3;
            else continue;
            sig = sig * 97 + static_cast<uint64_t>(owner * 24 + pv[c] + 1);
        }
        h = mix(h ^ (sig + 0x9E3779B97F4A7C15ull * (s + 1)));
    }
    return h;
}

}  // namespace

int suit_order(int card, int trump) { return kT.ord[trump][card]; }

Position Position::from_round(const Round& r) {
    Position p;
    for (int s = 0; s < 4; ++s) p.hands[s] = r.hands[s];
    for (int i = 0; i < 4; ++i) p.trick[i] = r.trick[i];
    p.n_in_trick = r.n_in_trick;
    p.leader = r.trick_first;
    p.trump = r.trump;
    p.tricks_done = r.n_tricks;
    p.won_mask = (r.tricks_won[0] ? 1 : 0) | (r.tricks_won[1] ? 2 : 0);
    return p;
}

DDSolver::DDSolver(int tt_bits) : tt_(size_t{1} << tt_bits), mask_(((uint64_t{1} << tt_bits) - 1) & ~uint64_t{1}) {
    for (auto& e : tt_) e = Entry{0, -kInf, kInf, -1, -1};
}

DDSolver::Entry* DDSolver::probe(uint64_t full) {
    Entry* b = &tt_[full & mask_];
    if (b[0].key == full) return &b[0];
    if (b[1].key == full) return &b[1];
    return nullptr;
}

void DDSolver::store(uint64_t full, int depth, int best, int alpha0, int beta0, int best_move) {
    Entry* b = &tt_[full & mask_];
    Entry* e = b[0].key == full ? &b[0] : b[1].key == full ? &b[1] : nullptr;
    Entry ne{full, static_cast<int16_t>(-kInf), static_cast<int16_t>(kInf), static_cast<int8_t>(best_move),
             static_cast<int8_t>(depth)};
    if (e) { ne.lo = e->lo; ne.hi = e->hi; }
    if (best <= alpha0) ne.hi = static_cast<int16_t>(best);
    else if (best >= beta0) ne.lo = static_cast<int16_t>(best);
    else ne.lo = ne.hi = static_cast<int16_t>(best);
    if (!e) e = depth >= b[0].depth ? &b[0] : &b[1];
    if (e == &b[0] && b[0].key != full && b[0].depth > b[1].depth) b[1] = b[0];  // demote old deep entry
    *e = ne;
}

// Legal moves with equivalent cards removed, best-first.
int DDSolver::order_moves(const Position& p, int seat, Cards legal, int tt_best, int* out) const {
    const int trump = p.trump;
    // Cards that can separate two of our cards: other hands + cards already in the trick.
    Cards sep = 0;
    for (int s = 0; s < 4; ++s)
        if (s != seat) sep |= p.hands[s];
    for (int i = 0; i < p.n_in_trick; ++i) sep |= bit(p.trick[i]);

    int n = 0;
    int keys[9];
    int win = -1, lead = -1;
    bool partner_winning = false;
    if (p.n_in_trick > 0) {
        lead = suit_of(p.trick[0]);
        int wpos = 0;
        for (int i = 1; i < p.n_in_trick; ++i)
            if (beats(p.trick[i], p.trick[wpos], lead, trump)) wpos = i;
        win = p.trick[wpos];
        partner_winning = (p.n_in_trick - wpos) == 2;
    }
    const auto& pts = kPoints.v[trump];

    for (int s = 0; s < 4; ++s) {
        const Cards in_suit = legal & kSuitMask[s];
        if (!in_suit) continue;
        const int* o = order_of(s, trump);
        bool group_open = false;
        int group_val = -1;
        bool master = true;  // no stronger card left in other hands
        for (int i = 0; i < 9; ++i) {
            const int c = s * 9 + o[i];
            if (sep & bit(c)) {
                group_open = false;
                if (p.hands[0] & bit(c) || p.hands[1] & bit(c) || p.hands[2] & bit(c) || p.hands[3] & bit(c))
                    master = false;
                continue;
            }
            if (!(in_suit & bit(c))) continue;
            if (group_open && pts[c] == group_val) continue;  // equivalent to the kept card
            group_open = true;
            group_val = pts[c];
            int k;
            if (c == tt_best) {
                k = 1 << 20;
            } else if (p.n_in_trick == 0) {
                k = master ? 2000 + pts[c] * 8 : 1000 - pts[c] * 8 - (is_trump(c, trump) ? 200 : 0);
            } else {
                const bool mine = beats(c, win, lead, trump);
                if (mine) k = 3000 - pts[c] * 4 - (is_trump(c, trump) && !is_trump(win, trump) ? 300 : 0);
                else if (partner_winning) k = 2500 + pts[c] * 8;
                else k = 1000 - pts[c] * 8;
                if (p.n_in_trick == 3 && mine) k += 1000;
            }
            out[n] = c;
            k = k * 64 + std::min(63, history_[p.n_in_trick][p.n_in_trick ? p.trick[0] : 35][c] >> 4);
            keys[n++] = k;
            master = false;
        }
    }
    for (int i = 1; i < n; ++i) {  // insertion sort, descending key
        const int c = out[i], k = keys[i];
        int j = i - 1;
        while (j >= 0 && keys[j] < k) { out[j + 1] = out[j]; keys[j + 1] = keys[j]; --j; }
        out[j + 1] = c;
        keys[j + 1] = k;
    }
    return n;
}

int DDSolver::search(Position& p, uint64_t key, int alpha, int beta) {
    ++nodes;
    if (p.tricks_done == 9) return p.won_mask == 1 ? kMatchBonus : p.won_mask == 2 ? -kMatchBonus : 0;
    if (p.tricks_done == 8) {
        // Last trick: everyone holds exactly one card, nothing to choose.
        int t[4], seat = p.leader;
        for (int i = 0; i < 4; ++i, seat = next_seat(seat)) t[i] = i < p.n_in_trick ? p.trick[i] : lowest(p.hands[seat]);
        int w = p.leader;
        for (int j = trick_winner_pos(t, p.trump); j > 0; --j) w = next_seat(w);
        const int pts = trick_points(t, p.trump, true);
        const int won = p.won_mask | 1 << team_of(w);
        return (team_of(w) == 0 ? pts : -pts) + (won == 1 ? kMatchBonus : won == 2 ? -kMatchBonus : 0);
    }

    uint64_t full = 0;
    int tt_best = -1;
    const int alpha0 = alpha, beta0 = beta;
    const bool trick_start = p.n_in_trick == 0;
    if (trick_start) {
        // Everything left goes to one side at most (plus a possible match bonus).
        const int rem = points_of(p.hands[0] | p.hands[1] | p.hands[2] | p.hands[3], p.trump) + kLastTrickBonus;
        const int ub = rem + ((p.won_mask & 2) ? 0 : kMatchBonus);
        const int lb = -rem - ((p.won_mask & 1) ? 0 : kMatchBonus);
        if (ub <= alpha) return ub;
        if (lb >= beta) return lb;
        full = abstract_key(p);
    } else {
        // Mid-trick: exact key (relative ranks are not safe next to the cards in the trick).
        // Caching these saves the within-trick work that MTD(f) repeats on every pass.
        full = key ^ kT.leader[p.leader] ^ kT.won[p.won_mask] ^ kT.trump[p.trump] ^ 0xA5A5A5A5DEADBEEFull;
        for (int i = 0; i < p.n_in_trick; ++i) full ^= kT.trick[i][p.trick[i]];
    }
    if (const Entry* e = probe(full)) {
        if (e->lo == e->hi) return e->lo;
        if (e->lo >= beta) return e->lo;
        if (e->hi <= alpha) return e->hi;
        alpha = std::max(alpha, static_cast<int>(e->lo));
        beta = std::min(beta, static_cast<int>(e->hi));
        tt_best = e->best;
    }

    const int seat = p.to_play();
    const bool maxi = team_of(seat) == 0;
    const Cards legal = legal_cards(p.hands[seat], p.trick, p.n_in_trick, p.trump);
    int moves[9];
    const int n = order_moves(p, seat, legal, tt_best, moves);

    int best = maxi ? -kInf : kInf, best_move = moves[0];
    for (int i = 0; i < n; ++i) {
        const int c = moves[i];
        p.hands[seat] &= ~bit(c);
        const uint64_t k2 = key ^ kT.card[seat][c];
        p.trick[p.n_in_trick++] = c;
        int v;
        if (p.n_in_trick < 4) {
            v = search(p, k2, alpha, beta);
        } else {
            const int pos = trick_winner_pos(p.trick, p.trump);
            int winner = p.leader;
            for (int j = 0; j < pos; ++j) winner = next_seat(winner);
            const int pts = trick_points(p.trick, p.trump, p.tricks_done == 8);
            const int delta = team_of(winner) == 0 ? pts : -pts;
            int saved[4];
            std::memcpy(saved, p.trick, sizeof saved);
            const int leader = p.leader, won = p.won_mask;
            p.leader = winner;
            p.n_in_trick = 0;
            p.tricks_done++;
            p.won_mask |= 1 << team_of(winner);
            v = delta + search(p, k2, alpha - delta, beta - delta);
            p.tricks_done--;
            p.leader = leader;
            p.won_mask = won;
            p.n_in_trick = 4;
            std::memcpy(p.trick, saved, sizeof saved);
        }
        p.n_in_trick--;
        p.hands[seat] |= bit(c);
        if (maxi) {
            if (v > best) { best = v; best_move = c; }
            if (best > alpha) alpha = best;
        } else {
            if (v < best) { best = v; best_move = c; }
            if (best < beta) beta = best;
        }
        if (alpha >= beta) {
            history_[p.n_in_trick][p.n_in_trick ? p.trick[0] : 35][c] += (9 - p.tricks_done) * (9 - p.tricks_done);
            break;
        }
    }

    store(full, 9 - p.tricks_done, best, alpha0, beta0, best_move);
    return best;
}

// MTD(f) with a switch to bisection when progress is slow: a sequence of null-window
// searches that converges on the exact value; bounds accumulate in the TT.
int DDSolver::mtdf(Position& p, uint64_t key, int guess) {
    Cards all = p.hands[0] | p.hands[1] | p.hands[2] | p.hands[3];
    for (int i = 0; i < p.n_in_trick; ++i) all |= bit(p.trick[i]);
    const int rem = points_of(all, p.trump) + kLastTrickBonus;
    int lo = -rem - ((p.won_mask & 1) ? 0 : kMatchBonus);
    int hi = rem + ((p.won_mask & 2) ? 0 : kMatchBonus);
    int g = std::min(std::max(guess, lo), hi);
    for (int iter = 0; lo < hi; ++iter) {
        int beta = iter < 4 ? (g == lo ? g + 1 : g) : lo + (hi - lo + 1) / 2;
        beta = std::min(std::max(beta, lo + 1), hi);
        g = search(p, key, beta - 1, beta);
        ++iterations;
        if (g < beta) hi = g; else lo = g;
    }
    return g;
}

int DDSolver::greedy_value(Position p) const {
    int v = 0;
    while (p.tricks_done < 9) {
        const int seat = p.to_play();
        int moves[9];
        order_moves(p, seat, legal_cards(p.hands[seat], p.trick, p.n_in_trick, p.trump), -1, moves);
        p.hands[seat] &= ~bit(moves[0]);
        p.trick[p.n_in_trick++] = moves[0];
        if (p.n_in_trick == 4) {
            int w = p.leader;
            for (int j = trick_winner_pos(p.trick, p.trump); j > 0; --j) w = next_seat(w);
            const int pts = trick_points(p.trick, p.trump, p.tricks_done == 8);
            v += team_of(w) == 0 ? pts : -pts;
            p.leader = w;
            p.n_in_trick = 0;
            p.tricks_done++;
            p.won_mask |= 1 << team_of(w);
        }
    }
    return v + (p.won_mask == 1 ? kMatchBonus : p.won_mask == 2 ? -kMatchBonus : 0);
}

static uint64_t hand_key(const Position& p) {
    uint64_t k = 0;
    for (int s = 0; s < 4; ++s)
        for (Cards h = p.hands[s]; h; h &= h - 1) k ^= kT.card[s][lowest(h)];
    return k;
}

int DDSolver::value(const Position& p) {
    Position q = p;
    return mtdf(q, hand_key(q), greedy_value(p));
}

int DDSolver::move_values(const Position& p, int* cards, int* values) {
    Position q = p;
    const int seat = q.to_play();
    const int sign = team_of(seat) == 0 ? 1 : -1;
    const Cards legal = legal_cards(q.hands[seat], q.trick, q.n_in_trick, q.trump);
    int reps[9];
    const int n_rep = order_moves(q, seat, legal, -1, reps);
    uint64_t key = hand_key(q);
    int rep_val[9];
    for (int i = 0; i < n_rep; ++i) {
        const int c = reps[i];
        q.hands[seat] &= ~bit(c);
        q.trick[q.n_in_trick++] = c;
        int v;
        if (q.n_in_trick < 4) {
            v = mtdf(q, key ^ kT.card[seat][c], greedy_value(q));
        } else {
            const int pos = trick_winner_pos(q.trick, q.trump);
            int winner = q.leader;
            for (int j = 0; j < pos; ++j) winner = next_seat(winner);
            const int pts = trick_points(q.trick, q.trump, q.tricks_done == 8);
            Position child = q;
            child.leader = winner;
            child.n_in_trick = 0;
            child.tricks_done++;
            child.won_mask |= 1 << team_of(winner);
            const int delta = team_of(winner) == 0 ? pts : -pts;
            v = delta + mtdf(child, key ^ kT.card[seat][c], greedy_value(child));
        }
        q.n_in_trick--;
        q.hands[seat] |= bit(c);
        rep_val[i] = v * sign;
    }
    // Every legal card gets the value of its representative (same suit, no separator between).
    int n = 0;
    for (Cards l = legal; l; l &= l - 1) {
        const int c = lowest(l);
        int best_i = 0, best_d = 99;
        for (int i = 0; i < n_rep; ++i) {
            if (suit_of(reps[i]) != suit_of(c)) continue;
            const int d = std::abs(kT.ord[q.trump][reps[i]] - kT.ord[q.trump][c]);
            if (d < best_d && kPoints.v[q.trump][reps[i]] == kPoints.v[q.trump][c]) {
                bool separated = false;
                const int lo = std::min(kT.ord[q.trump][reps[i]], kT.ord[q.trump][c]);
                const int hi = std::max(kT.ord[q.trump][reps[i]], kT.ord[q.trump][c]);
                const int* o = order_of(suit_of(c), q.trump);
                for (int k = lo + 1; k < hi; ++k) {
                    const int x = suit_of(c) * 9 + o[k];
                    for (int s = 0; s < 4; ++s)
                        if (s != seat && (q.hands[s] & bit(x))) separated = true;
                    for (int t = 0; t < q.n_in_trick; ++t)
                        if (q.trick[t] == x) separated = true;
                }
                if (!separated) { best_d = d; best_i = i; }
            }
        }
        cards[n] = c;
        values[n++] = rep_val[best_i];
    }
    return n;
}

}  // namespace jass
