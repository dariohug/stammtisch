// C ABI for Python (ctypes). All arrays are caller-owned, C-contiguous.
#include <cstring>
#include <new>

#include <cmath>
#include <memory>
#include <thread>
#include <vector>

#include "agents.hpp"
#include "dd.hpp"
#include "features.hpp"
#include "jass.hpp"
#include "net.hpp"
#include "pimc.hpp"

using namespace jass;

template <typename F>
static void parallel_for(int64_t n, int32_t threads, F f) {
    threads = std::max(1, threads);
    std::vector<std::thread> pool;
    for (int t = 0; t < threads; ++t)
        pool.emplace_back([=] {
            for (int64_t j = n * t / threads; j < n * (t + 1) / threads; ++j) f(j);
        });
    for (auto& th : pool) th.join();
}

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

static Rules rules_preset(int32_t preset) { return preset == 1 ? Rules::plain() : Rules::swisslos(); }

void jass_env_set_rules(void* p, int32_t preset) {
    auto* e = static_cast<Env*>(p);
    for (int i = 0; i < e->n; ++i) e->rounds[i].rules = rules_preset(preset);
}

// Agents by spec ("random", "heuristic", "pimc:32", ...). Returns nullptr on a bad spec.
void* jass_agent_create(const char* spec) { return make_agent(spec).release(); }
void jass_agent_destroy(void* a) { delete static_cast<Agent*>(a); }

// Lets an agent act for the given envs (mask[i] != 0), e.g. as opponents.
void jass_env_agent_actions(void* p, void* agent, const int8_t* mask, int16_t* actions) {
    auto* e = static_cast<Env*>(p);
    auto* a = static_cast<Agent*>(agent);
    for (int i = 0; i < e->n; ++i) {
        if (!mask[i]) continue;
        const Round& r = e->rounds[i];
        if (r.trump < 0) {
            const int t = a->choose_trump(r, e->rng);
            actions[i] = static_cast<int16_t>(t == kPush ? kActPush : kActTrump + t);
        } else {
            actions[i] = static_cast<int16_t>(a->choose_card(r, e->rng));
        }
    }
}

// ---------------------------------------------------------------------------
// Game analysis
// ---------------------------------------------------------------------------

static void set_belief(SearchConfig& cfg, double belief) {
    if (belief <= 0) return;
    auto card = Mlp::load(models_dir() + "/card_policy.bin");
    auto trump = Mlp::load(models_dir() + "/trump_policy.bin");
    if (!card || !trump) return;
    cfg.card_model = card.get();
    cfg.trump_model = trump.get();
    cfg.belief = belief;
}

// Seats of each played card and the four starting hands, from the play order alone.
// Returns false if the cards are not distinct.
static bool reconstruct(const int8_t* cards, int n_cards, int dealer, int trump, int* seats, Cards hands[4]) {
    Cards seen = 0;
    for (int s = 0; s < 4; ++s) hands[s] = 0;
    int first = next_seat(dealer);
    for (int t = 0; t * 4 < n_cards; ++t) {
        int trick[4], seat = first;
        for (int i = 0; i < 4 && t * 4 + i < n_cards; ++i, seat = next_seat(seat)) {
            const int c = cards[t * 4 + i];
            if (c < 0 || c >= 36 || (seen & bit(c))) return false;
            seen |= bit(c);
            hands[seat] |= bit(c);
            seats[t * 4 + i] = seat;
            trick[i] = c;
        }
        if (t * 4 + 4 <= n_cards) {
            const int pos = trick_winner_pos(trick, trump);
            for (int j = 0; j < pos; ++j) first = next_seat(first);
        }
    }
    return true;
}

// Analyses a recorded round. `hands_in` may be null when all 36 cards are given
// (the hands follow from the play order). For each played card k whose seat is in
// seat_mask, out_legal[k] is the legal set and out_info / out_dd[k*36 + c] the value of
// playing c instead: PIMC mean from that player's information, and the double-dummy
// value with all cards known. Values: own team minus opponents, raw card points of the
// remaining play (incl. the current trick). Others are NaN.
// Returns 0, -1 for inconsistent input, or k+1 if card k was illegal.
int32_t jass_analyze_round(const int8_t* cards, int32_t n_cards, const uint64_t* hands_in, int32_t dealer,
                           int32_t trump, int32_t pushed, int32_t rules, int32_t seat_mask, int32_t samples,
                           int32_t exact_left, void* rollout, double belief, uint64_t seed, uint64_t* out_legal,
                           float* out_info, float* out_dd, float* out_se) {
    int seats[36];
    Cards hands[4];
    if (!reconstruct(cards, n_cards, dealer, trump, seats, hands)) return -1;
    if (hands_in) {
        for (int s = 0; s < 4; ++s) {
            if ((hands[s] & ~hands_in[s]) || popcount(hands_in[s]) != 9) return -1;
            hands[s] = hands_in[s];
        }
    } else if (n_cards != 36) {
        return -1;
    }
    Round r;
    r.reset(hands, dealer, rules_preset(rules));
    if (pushed) r.choose_trump(kPush);
    r.choose_trump(trump);
    Rng rng(seed);
    DDSolver dd(21);
    SearchConfig cfg;
    cfg.samples = samples;
    cfg.exact_left = exact_left;
    cfg.rollout = static_cast<Agent*>(rollout);
    set_belief(cfg, belief);
    for (int k = 0; k < 36 * 36; ++k) out_info[k] = out_dd[k] = out_se[k] = NAN;
    for (int k = 0; k < n_cards; ++k) {
        const int c = cards[k];
        out_legal[k] = 0;
        if (!(r.legal() & bit(c))) return k + 1;
        if (seat_mask >> r.to_play & 1) {
            out_legal[k] = r.legal();
            // Hindsight: the true deal, evaluated like one PIMC sample (exact when affordable).
            {
                int pc[9];
                double pv[9];
                const int n = deal_card_values(r, cfg, rng, dd, pc, pv);
                for (int j = 0; j < n; ++j) out_dd[k * 36 + pc[j]] = static_cast<float>(pv[j]);
            }
            if (samples > 0) {
                int pc[9];
                double pv[9], se[9];
                const int n = pimc_card_values(r, cfg, rng, dd, pc, pv, se);
                if (n == 1) pv[0] = out_dd[k * 36 + pc[0]];  // forced card: no choice to evaluate
                for (int j = 0; j < n; ++j) {
                    out_info[k * 36 + pc[j]] = static_cast<float>(pv[j]);
                    out_se[k * 36 + pc[j]] = static_cast<float>(se[j]);
                }
            }
        }
        r.play(c);
    }
    return 0;
}

// Trump decision analysis for the chooser holding `hand`. out_info[0..6]: mean slate
// difference over `samples` random deals (modes 0..5, push at 6; see pimc_trump_values).
// If hands_in is given, out_dd[0..5] is the double-dummy slate difference of each mode
// for the actual deal (what perfect play would have yielded).
void jass_analyze_trump(uint64_t hand, const uint64_t* hands_in, int32_t dealer, int32_t pushed, int32_t rules,
                        int32_t samples, int32_t exact_left, void* rollout, double belief, uint64_t seed,
                        double* out_info, double* out_dd) {
    Round r;
    Cards h[4] = {0, 0, 0, 0};
    r.reset(h, dealer, rules_preset(rules));
    r.pushed = pushed != 0;
    const int me = r.trump_chooser();
    r.hands[me] = hand;
    Rng rng(seed);
    DDSolver dd(21);
    SearchConfig cfg;
    cfg.samples = samples;
    cfg.exact_left = exact_left;
    cfg.rollout = static_cast<Agent*>(rollout);
    set_belief(cfg, belief);
    pimc_trump_values(r, cfg, rng, dd, out_info, cfg.rollout);
    if (!hands_in) return;
    const int team = team_of(me);
    for (int m = 0; m < 6; ++m) {
        Round t;
        t.reset(hands_in, dealer, rules_preset(rules));
        t.pushed = pushed != 0;
        t.choose_trump(m);
        const int d = evaluate_deal(t, cfg, rng, dd);
        out_dd[m] = t.rules.mult[m] * ((team == 0 ? d : -d) + t.bonus[team] - t.bonus[1 - team]);
    }
}

// Advice at the table: only my hand and the public play are known. `played` are the cards
// so far in play order (first card by the forehand). Returns the number of legal cards
// (values: expected raw point difference of the remaining play for my team), -1 for
// inconsistent input, -2 if it is not my turn.
int32_t jass_advise(uint64_t my_hand, const int8_t* played, int32_t n_played, int32_t me, int32_t dealer,
                    int32_t trump, int32_t pushed, int32_t rules, int32_t samples, int32_t exact_left, void* rollout,
                    double belief, uint64_t seed, int32_t* out_cards, double* out_values, double* out_se) {
    int seats[36];
    Cards partial[4];
    if (n_played < 0 || n_played > 35 || !reconstruct(played, n_played, dealer, trump, seats, partial)) return -1;
    if (partial[me] & my_hand) return -1;
    Cards hands[4];
    for (int s = 0; s < 4; ++s) hands[s] = partial[s];
    hands[me] |= my_hand;
    if (popcount(hands[me]) != 9) return -1;
    Round r;
    r.reset(hands, dealer, rules_preset(rules));
    if (pushed) r.choose_trump(kPush);
    r.choose_trump(trump);
    for (int s = 0; s < 4; ++s)
        if (s != me) r.shown[s] = 0;  // others' Weis are unknown here
    for (int k = 0; k < n_played; ++k) r.play(played[k]);
    if (r.to_play != me) return -2;
    Rng rng(seed);
    DDSolver dd(21);
    SearchConfig cfg;
    cfg.samples = samples;
    cfg.exact_left = exact_left;
    cfg.rollout = static_cast<Agent*>(rollout);
    set_belief(cfg, belief);
    int cards[9];
    double values[9], se[9];
    const int n = pimc_card_values(r, cfg, rng, dd, cards, values, se);
    for (int i = 0; i < n; ++i) { out_cards[i] = cards[i]; out_values[i] = values[i]; out_se[i] = se[i]; }
    return n;
}

// An agent acting from a player's observation (my hand + public play), e.g. for jass-kit.
// Returns the card, or -1 for inconsistent input / -2 if it is not `me`'s turn.
int32_t jass_agent_card(void* agent, uint64_t my_hand, const int8_t* played, int32_t n_played, int32_t me,
                        int32_t dealer, int32_t trump, int32_t pushed, int32_t rules, uint64_t seed) {
    int seats[36];
    Cards hands[4];
    if (n_played < 0 || n_played > 35 || !reconstruct(played, n_played, dealer, trump, seats, hands)) return -1;
    if (hands[me] & my_hand) return -1;
    hands[me] |= my_hand;
    if (popcount(hands[me]) != 9) return -1;
    Round r;
    r.reset(hands, dealer, rules_preset(rules));
    if (pushed) r.choose_trump(kPush);
    r.choose_trump(trump);
    for (int s = 0; s < 4; ++s)
        if (s != me) r.shown[s] = 0;
    for (int k = 0; k < n_played; ++k) r.play(played[k]);
    if (r.to_play != me) return -2;
    Rng rng(seed);
    return static_cast<Agent*>(agent)->choose_card(r, rng);
}

// Trump call of an agent holding `hand` (the chooser follows from dealer/pushed).
// Returns 0..5, or 10 (kPush).
int32_t jass_agent_trump(void* agent, uint64_t hand, int32_t dealer, int32_t pushed, int32_t rules, uint64_t seed) {
    Round r;
    Cards h[4] = {0, 0, 0, 0};
    r.reset(h, dealer, rules_preset(rules));
    r.pushed = pushed != 0;
    r.hands[r.trump_chooser()] = hand;
    Rng rng(seed);
    return static_cast<Agent*>(agent)->choose_trump(r, rng);
}

// Human trump policy: probabilities of the 6 modes and push (out[7]); returns 0, or -1
// if the model is missing.
int32_t jass_trump_policy(uint64_t hand, int32_t pushed, double* out) {
    auto m = Mlp::load(models_dir() + "/trump_policy.bin");
    if (!m) return -1;
    float x[kTrumpFeatures], o[kTrumpActions];
    trump_features(hand, pushed != 0, x);
    m->forward(x, o);
    const int n = pushed ? 6 : 7;
    float mx = o[0];
    for (int i = 1; i < n; ++i) mx = std::max(mx, o[i]);
    double total = 0;
    for (int i = 0; i < 7; ++i) total += out[i] = i < n ? std::exp(static_cast<double>(o[i] - mx)) : 0.0;
    for (int i = 0; i < 7; ++i) out[i] /= total;
    return 0;
}

// Weis / Stöck of a full deal for a trump: bonus[2] per team, weis_team (-1 none), shown[4].
void jass_bonus(const uint64_t* hands, int32_t dealer, int32_t trump, int32_t rules, int32_t* bonus,
                int32_t* weis_team, uint64_t* shown) {
    Round r;
    r.reset(hands, dealer, rules_preset(rules));
    r.choose_trump(trump);
    bonus[0] = r.bonus[0];
    bonus[1] = r.bonus[1];
    *weis_team = r.weis_team;
    for (int s = 0; s < 4; ++s) shown[s] = r.shown[s];
}

// Double-dummy value (team 0 minus team 1, raw points of the remaining play) of a
// position given all hands; for tests and tools.
int32_t jass_dd_value(const uint64_t* hands, const int32_t* trick, int32_t n_in_trick, int32_t leader,
                      int32_t trump, int32_t tricks_done, int32_t won_mask) {
    Position p;
    for (int s = 0; s < 4; ++s) p.hands[s] = hands[s];
    for (int i = 0; i < 4; ++i) p.trick[i] = i < n_in_trick ? trick[i] : -1;
    p.n_in_trick = n_in_trick;
    p.leader = leader;
    p.trump = trump;
    p.tricks_done = tricks_done;
    p.won_mask = won_mask;
    static thread_local DDSolver dd(18);
    return dd.value(p);
}

// ---------------------------------------------------------------------------
// Training samples from logged games (generated on the fly, multi-threaded)
// ---------------------------------------------------------------------------
int32_t jass_card_features_size() { return kCardFeatures; }
int32_t jass_trump_features_size() { return kTrumpFeatures; }

static void logged_round(const int8_t* cards, int dealer, int trump, int pushed, Round& r) {
    int seats[36];
    Cards hands[4];
    reconstruct(cards, 36, dealer, trump, seats, hands);
    r.reset(hands, dealer, Rules::plain());
    if (pushed) r.choose_trump(kPush);
    r.choose_trump(trump);
}

// Card samples: game games[j], decision pos[j] (0..35). X[j*kCardFeatures], legal set,
// label = the card the human played.
void jass_log_card_samples(int64_t n, const int32_t* games, const int8_t* pos, const int8_t* cards,
                           const int8_t* dealer, const int8_t* trump, const int8_t* forehand, float* X,
                           uint64_t* legal, int8_t* label, int32_t threads) {
    parallel_for(n, threads, [&](int64_t j) {
        const int g = games[j];
        const int8_t* c = cards + static_cast<int64_t>(g) * 36;
        Round r;
        logged_round(c, dealer[g], trump[g], forehand[g] == 0, r);
        for (int k = 0; k < pos[j]; ++k) r.play(c[k]);
        card_features(r, X + j * kCardFeatures);
        legal[j] = r.legal();
        label[j] = c[pos[j]];
    });
}

// Trump samples: which[j] = 0 forehand's decision (label: trump or 6 = push),
// 1 = partner's decision after a push (label: trump).
void jass_log_trump_samples(int64_t n, const int32_t* games, const int8_t* which, const int8_t* cards,
                            const int8_t* dealer, const int8_t* trump, const int8_t* forehand, float* X,
                            int8_t* label, int32_t threads) {
    parallel_for(n, threads, [&](int64_t j) {
        const int g = games[j];
        int seats[36];
        Cards hands[4];
        reconstruct(cards + static_cast<int64_t>(g) * 36, 36, dealer[g], trump[g], seats, hands);
        const int fh = next_seat(dealer[g]);
        if (which[j] == 0) {
            trump_features(hands[fh], false, X + j * kTrumpFeatures);
            label[j] = static_cast<int8_t>(forehand[g] == 0 ? 6 : trump[g]);
        } else {
            trump_features(hands[partner(fh)], true, X + j * kTrumpFeatures);
            label[j] = trump[g];
        }
    });
}

// Features of the acting player in env i (for learning agents / tools).
void jass_env_card_features(void* p, int32_t i, float* X) {
    card_features(static_cast<Env*>(p)->rounds[i], X);
}

}  // extern "C"
