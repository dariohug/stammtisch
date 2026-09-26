// C ABI for Python (ctypes). All arrays are caller-owned, C-contiguous.
#include <cstring>
#include <new>

#include "agents.hpp"
#include "jass.hpp"

using namespace jass;

extern "C" {

int jass_abi_version() { return 1; }

uint64_t jass_legal(uint64_t hand, const int32_t* trick, int32_t n, int32_t trump) {
    int t[4];
    for (int i = 0; i < n; ++i) t[i] = trick[i];
    return legal_cards(hand, t, n, trump);
}

int32_t jass_trick_winner_pos(const int32_t* trick, int32_t trump) {
    int t[4] = {trick[0], trick[1], trick[2], trick[3]};
    return trick_winner_pos(t, trump);
}

// Replays logged games and checks them against the rules.
//   cards[g*36 + i]  i-th card played (trick-major), first[g*9 + k] first seat of trick k,
//   win[g*9 + k] / points[g*9 + k] logged winner / points, trump[g], dealer[g].
// status[g]: 0 ok, 1 not a permutation of the deck, 2 illegal card, 3 wrong first seat,
//            4 winner mismatch, 5 points mismatch.  err_pos[g]: index of first bad card.
int64_t jass_replay_check(int64_t n_games, const int8_t* cards, const int8_t* first, const int8_t* win,
                          const int16_t* points, const int8_t* trump, const int8_t* dealer,
                          int8_t* status, int8_t* err_pos) {
    int64_t n_bad = 0;
    for (int64_t g = 0; g < n_games; ++g) {
        const int8_t* c = cards + g * 36;
        const int8_t* f = first + g * 9;
        const int8_t* w = win + g * 9;
        const int16_t* pts = points + g * 9;
        int8_t st = 0, pos = -1;
        Cards hands[4] = {0, 0, 0, 0}, seen = 0;
        for (int k = 0; k < 9 && !st; ++k) {
            int seat = f[k];
            for (int i = 0; i < 4; ++i, seat = next_seat(seat)) {
                const int card = c[k * 4 + i];
                if (card < 0 || card >= 36 || (seen & bit(card))) { st = 1; pos = static_cast<int8_t>(k * 4 + i); break; }
                seen |= bit(card);
                hands[seat] |= bit(card);
            }
        }
        if (!st) {
            Round r;
            r.reset(hands, dealer[g]);
            r.trump = trump[g];
            for (int k = 0; k < 9 && !st; ++k) {
                if (r.to_play != f[k]) { st = 3; pos = static_cast<int8_t>(k * 4); break; }
                const int before0 = r.points[0], before1 = r.points[1];
                for (int i = 0; i < 4; ++i) {
                    const int card = c[k * 4 + i];
                    if (!(r.legal() & bit(card))) { st = 2; pos = static_cast<int8_t>(k * 4 + i); break; }
                    r.play(card);
                }
                if (st) break;
                if (r.winners[k] != w[k]) { st = 4; pos = static_cast<int8_t>(k * 4 + 3); break; }
                if (r.points[0] - before0 + r.points[1] - before1 != pts[k]) { st = 5; pos = static_cast<int8_t>(k * 4 + 3); }
            }
        }
        status[g] = st;
        err_pos[g] = pos;
        n_bad += st != 0;
    }
    return n_bad;
}

// ---------------------------------------------------------------------------
// Vectorised environment for learning agents.
// Actions: 0..35 play card, 36..41 declare trump 0..5, 42 push.
// ---------------------------------------------------------------------------
struct Env {
    int n;
    uint64_t seed;
    uint64_t counter;
    Rng rng;
    Round* rounds;
};

constexpr int kActTrump = 36, kActPush = 42;

void* jass_env_create(int32_t n, uint64_t seed) {
    auto* e = new (std::nothrow) Env{n, seed, 0, Rng(seed), new (std::nothrow) Round[static_cast<size_t>(n)]};
    return e;
}

void jass_env_destroy(void* p) {
    auto* e = static_cast<Env*>(p);
    delete[] e->rounds;
    delete e;
}

static void env_reset_one(Env* e, int i) {
    Cards h[4];
    deal(e->rng, h);
    e->rounds[i].reset(h, static_cast<int>(e->counter++ & 3));
}

void jass_env_reset(void* p) {
    auto* e = static_cast<Env*>(p);
    for (int i = 0; i < e->n; ++i) env_reset_one(e, i);
}

// Observation (per env i):
//   seat[i]        acting seat
//   legal[i]       action mask as bits 0..42
//   hands[i*4+s]   hand of every seat (the caller decides what the agent may see)
//   history[i*36]  cards in play order, -1 padded
//   info[i*8]      trump, declarer, pushed, dealer, n_played, trick_first, points0, points1
void jass_env_observe(void* p, int8_t* seat, uint64_t* legal, uint64_t* hands, int8_t* history, int16_t* info) {
    auto* e = static_cast<Env*>(p);
    for (int i = 0; i < e->n; ++i) {
        const Round& r = e->rounds[i];
        const int np = r.n_tricks * 4 + r.n_in_trick;
        if (r.trump < 0) {
            seat[i] = static_cast<int8_t>(r.trump_chooser());
            legal[i] = (Cards{0x3F} << kActTrump) | (r.pushed ? 0 : Cards{1} << kActPush);
        } else {
            seat[i] = static_cast<int8_t>(r.to_play);
            legal[i] = r.legal();
        }
        for (int s = 0; s < 4; ++s) hands[i * 4 + s] = r.hands[s];
        for (int k = 0; k < 36; ++k) history[i * 36 + k] = static_cast<int8_t>(k < np ? r.history[k] : -1);
        int16_t* inf = info + i * 8;
        inf[0] = static_cast<int16_t>(r.trump); inf[1] = static_cast<int16_t>(r.declarer);
        inf[2] = r.pushed; inf[3] = static_cast<int16_t>(r.dealer); inf[4] = static_cast<int16_t>(np);
        inf[5] = static_cast<int16_t>(r.trick_first);
        inf[6] = static_cast<int16_t>(r.points[0]); inf[7] = static_cast<int16_t>(r.points[1]);
    }
}

// Applies one action per env. Finished rounds report final scores in `final_score`
// (team 0, team 1, match bonus included) and are re-dealt when auto_reset != 0.
// Returns the number of illegal actions (those envs are left unchanged).
int32_t jass_env_step(void* p, const int16_t* actions, int32_t auto_reset, int8_t* done, int16_t* final_score) {
    auto* e = static_cast<Env*>(p);
    int32_t illegal = 0;
    for (int i = 0; i < e->n; ++i) {
        Round& r = e->rounds[i];
        const int a = actions[i];
        done[i] = 0;
        if (r.trump < 0) {
            if (a == kActPush && !r.pushed) r.choose_trump(kPush);
            else if (a >= kActTrump && a < kActTrump + 6) r.choose_trump(a - kActTrump);
            else { ++illegal; continue; }
        } else {
            if (a < 0 || a >= 36 || !(r.legal() & bit(a))) { ++illegal; continue; }
            r.play(a);
        }
        if (r.done()) {
            done[i] = 1;
            final_score[i * 2] = static_cast<int16_t>(r.score(0));
            final_score[i * 2 + 1] = static_cast<int16_t>(r.score(1));
            if (auto_reset) env_reset_one(e, i);
        }
    }
    return illegal;
}

// Lets built-in agents act for the given envs (mask[i] != 0), e.g. as opponents.
void jass_env_builtin_actions(void* p, int32_t agent, const int8_t* mask, int16_t* actions) {
    auto* e = static_cast<Env*>(p);
    const auto k = agent == 0 ? AgentKind::Random : AgentKind::Heuristic;
    for (int i = 0; i < e->n; ++i) {
        if (!mask[i]) continue;
        const Round& r = e->rounds[i];
        if (r.trump < 0) {
            const int t = choose_trump(k, r, e->rng);
            actions[i] = static_cast<int16_t>(t == kPush ? kActPush : kActTrump + t);
        } else {
            actions[i] = static_cast<int16_t>(choose_card(k, r, e->rng));
        }
    }
}

}  // extern "C"
