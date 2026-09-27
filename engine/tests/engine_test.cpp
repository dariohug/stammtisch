// Engine self-tests: DD solver vs. brute-force minimax, Weis scoring, sampler constraints.
// Also prints solver timings.  Run: build/engine-test
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>

#include "dd.hpp"
#include "jass.hpp"
#include "pimc.hpp"

using namespace jass;

static int failures = 0;
#define CHECK(cond, ...)                                  \
    do {                                                  \
        if (!(cond)) {                                    \
            ++failures;                                   \
            std::printf("FAIL %s:%d: ", __FILE__, __LINE__); \
            std::printf(__VA_ARGS__);                     \
            std::printf("\n");                            \
        }                                                 \
    } while (0)

// Plain minimax, no pruning: the reference.
static int brute(Position& p) {
    if (p.tricks_done == 9) return p.won_mask == 1 ? kMatchBonus : p.won_mask == 2 ? -kMatchBonus : 0;
    const int seat = p.to_play();
    const bool maxi = team_of(seat) == 0;
    int best = maxi ? -10000 : 10000;
    for (Cards l = legal_cards(p.hands[seat], p.trick, p.n_in_trick, p.trump); l; l &= l - 1) {
        const int c = lowest(l);
        Position q = p;
        q.hands[seat] &= ~bit(c);
        q.trick[q.n_in_trick++] = c;
        int v = 0;
        if (q.n_in_trick == 4) {
            const int pos = trick_winner_pos(q.trick, q.trump);
            int w = q.leader;
            for (int j = 0; j < pos; ++j) w = next_seat(w);
            const int pts = trick_points(q.trick, q.trump, q.tricks_done == 8);
            v = team_of(w) == 0 ? pts : -pts;
            q.leader = w;
            q.n_in_trick = 0;
            q.tricks_done++;
            q.won_mask |= 1 << team_of(w);
        }
        v += brute(q);
        best = maxi ? std::max(best, v) : std::min(best, v);
    }
    return best;
}

// Random position with `left` tricks to play and `in_trick` cards already in the trick.
static Position random_position(Rng& rng, int left, int in_trick) {
    Cards h[4];
    deal(rng, h);
    Round r;
    r.reset(h, static_cast<int>(rng.below(4)), Rules::plain());
    r.choose_trump(static_cast<int>(rng.below(6)));
    while (r.n_played() < (9 - left) * 4 + in_trick) r.play(rng.pick(r.legal()));
    return Position::from_round(r);
}

static void test_dd_vs_brute() {
    Rng rng(42);
    DDSolver dd(16);
    int n = 0;
    for (int left = 1; left <= 3; ++left)  // brute force explodes beyond 3 tricks
        for (int in_trick = 0; in_trick < 4; ++in_trick)
            for (int i = 0; i < (left <= 2 ? 300 : 40); ++i, ++n) {
                Position p = random_position(rng, left, in_trick);
                Position q = p;
                const int ref = brute(q);
                const int got = dd.value(p);
                CHECK(ref == got, "left=%d in_trick=%d: dd=%d brute=%d", left, in_trick, got, ref);
                int cards[9], vals[9];
                const int m = dd.move_values(p, cards, vals);
                int best = -10000;
                for (int k = 0; k < m; ++k) best = std::max(best, vals[k]);
                const int mover_ref = team_of(p.to_play()) == 0 ? ref : -ref;
                CHECK(best == mover_ref, "move_values max %d != value %d", best, mover_ref);
                CHECK(m == popcount(legal_cards(p.hands[p.to_play()], p.trick, p.n_in_trick, p.trump)),
                      "move_values must cover every legal card");
            }
    std::printf("dd vs brute force: %d positions\n", n);
}

static void test_weis() {
    auto hand = [](std::initializer_list<const char*> names) {
        Cards h = 0;
        for (const char* nm : names) {
            const char* suits = "DHSC";
            const char* ranks[9] = {"A", "K", "Q", "J", "10", "9", "8", "7", "6"};
            int s = static_cast<int>(std::strchr(suits, nm[0]) - suits);
            for (int r = 0; r < 9; ++r)
                if (std::strcmp(nm + 1, ranks[r]) == 0) h |= bit(s * 9 + r);
        }
        return h;
    };
    CHECK(weis_of(hand({"HA", "HK", "HQ"}), 0).points == 20, "Dreiblatt");
    CHECK(weis_of(hand({"H9", "H8", "H7", "H6"}), 0).points == 50, "Vierblatt");
    CHECK(weis_of(hand({"SA", "SK", "SQ", "SJ", "S10"}), 0).points == 100, "Fuenfblatt");
    CHECK(weis_of(hand({"DJ", "HJ", "SJ", "CJ"}), 0).points == 200, "vier Puure");
    CHECK(weis_of(hand({"D9", "H9", "S9", "C9"}), 0).points == 150, "vier Nell");
    CHECK(weis_of(hand({"D6", "H6", "S6", "C6"}), 0).points == 0, "vier Sechser zaehlen nicht");
    CHECK(weis_of(hand({"HA", "HK", "HQ", "SA", "SK", "SQ"}), 0).points == 40, "zwei Dreiblatt");
    // Tie: same points and cards, trump sequence wins.
    CHECK(weis_of(hand({"HA", "HK", "HQ"}), 1).best_key > weis_of(hand({"SA", "SK", "SQ"}), 1).best_key,
          "Trumpfweis gewinnt");
    // Stöck: K+Q of trump (Herz) in seat 2's hand.
    const Cards kq = bit(1 * 9 + 1) | bit(1 * 9 + 2);
    Round r;
    Cards hh[4] = {0, 0, 0, 0};
    hh[2] = kq;
    r.reset(hh, 0, Rules::swisslos());
    r.choose_trump(1);
    CHECK(r.bonus[0] == kStoeck && r.bonus[1] == 0, "Stoeck for team of seat 2");
    r.reset(hh, 0, Rules::swisslos());
    r.choose_trump(4);
    CHECK(r.bonus[0] == 0, "no Stoeck in Obenabe");
}

static void test_sampler() {
    Rng rng(7);
    int checked = 0;
    for (int g = 0; g < 200; ++g) {
        Cards h[4];
        deal(rng, h);
        Round r;
        r.reset(h, g & 3, Rules::swisslos());
        r.choose_trump(static_cast<int>(rng.below(6)));
        const int stop = static_cast<int>(rng.below(33));
        while (r.n_played() < stop) r.play(rng.pick(r.legal()));
        const int me = r.to_play;
        const InfoSet is = info_set(r, me);
        // The true deal must satisfy the inferred constraints.
        for (int s = 0; s < 4; ++s)
            if (s != me) {
                CHECK((r.hands[s] & ~is.allowed[s] & ~is.fixed[s]) == 0, "true hand violates void inference");
                CHECK(popcount(r.hands[s]) == is.need[s] + popcount(is.fixed[s]), "hand size");
            }
        for (int k = 0; k < 20; ++k) {
            Cards out[4];
            sample_deal(is, rng, out);
            CHECK(out[me] == r.hands[me], "own hand fixed");
            Cards u = 0;
            for (int s = 0; s < 4; ++s) {
                CHECK(popcount(out[s]) == popcount(r.hands[s]), "sample hand size");
                CHECK((u & out[s]) == 0, "sample overlap");
                u |= out[s];
            }
            CHECK((u | r.played) == kAll, "sample covers all unseen cards");
            ++checked;
        }
    }
    std::printf("sampler: %d samples checked\n", checked);
}

static void bench_depth() {
    Rng rng(11);
    DDSolver dd(20);
    for (int left = 4; left <= 8; ++left) {
        double total = 0, worst = 0;
        uint64_t nodes = 0;
        int n = 0;
        auto start = std::chrono::steady_clock::now();
        while (n < 20 && std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() < 20) {
            Position p = random_position(rng, left, 0);
            dd.nodes = 0;
            dd.iterations = 0;
            auto t0 = std::chrono::steady_clock::now();
            const int v = dd.value(p);
            if (left == 6) std::printf("   v=%d iters=%llu nodes=%llu\n", v, (unsigned long long)dd.iterations, (unsigned long long)dd.nodes);
            const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            total += ms; worst = std::max(worst, ms); nodes += dd.nodes; ++n;
        }
        std::printf("tricks left %d: mean %.2f ms, worst %.2f ms, %.0f nodes (n=%d)\n", left, total / n, worst,
                    static_cast<double>(nodes) / n, n);
    }
}

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IOLBF, 0);
    if (argc > 1 && std::strcmp(argv[1], "--bench") == 0) { bench_depth(); return 0; }
    test_weis();
    test_dd_vs_brute();
    test_sampler();
    std::printf(failures ? "%d FAILURES\n" : "all engine tests passed\n", failures);
    return failures ? 1 : 0;
}
