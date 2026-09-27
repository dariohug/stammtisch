// Perfect Information Monte Carlo: sample the hidden hands consistently with what a
// player has seen, solve each sample double-dummy, average.
#pragma once
#include <array>
#include <vector>

#include "dd.hpp"
#include "jass.hpp"

namespace jass {

// What seat `me` knows: own hand, public history, shown Weis, inferred voids.
struct InfoSet {
    int me;
    Cards pool;        // unseen cards, to be distributed among the other seats
    Cards fixed[4];    // cards known to be in a hand (own hand, shown Weis)
    Cards allowed[4];  // cards a seat may still hold (voids removed)
    int need[4];       // pool cards each seat receives
};

InfoSet info_set(const Round& r, int me);
// Fills out[4] with a deal consistent with the info set. Returns false only if the
// constraints could not be met (then constraints on voids were relaxed).
bool sample_deal(const InfoSet& is, Rng& rng, Cards out[4]);

class Agent;
class Mlp;

// How a sampled deal is evaluated: exactly (double dummy) once at most `exact_left`
// tricks remain; before that every player follows the rollout policy (their own view
// only) until the exact part starts. rollout == nullptr means the heuristic policy.
struct SearchConfig {
    int samples = 32;
    int exact_left = 5;
    Agent* rollout = nullptr;
    // Belief: weight sampled deals by how likely the other players' observed actions
    // (trump call, push, every card) are under these behaviour models, raised to `belief`
    // (0 = uniform sampling, 1 = full Bayesian weighting). `candidates` deals are drawn per
    // evaluated sample and resampled by weight.
    const Mlp* card_model = nullptr;
    const Mlp* trump_model = nullptr;
    double belief = 0.0;
    int candidates = 4;
};

// log P(observed actions of everyone except `me` | deal), under the behaviour models.
double action_log_likelihood(const Round& r, int me, const Cards current[4], const SearchConfig& cfg);
// Draws cfg.samples deals for the player to move, belief-weighted if configured.
void draw_deals(const Round& r, int me, const SearchConfig& cfg, Rng& rng, std::vector<std::array<Cards, 4>>& out);

// Mean value of each legal card of the player to move (raw point difference of the
// remaining play, own team minus opponents). Returns the number of legal cards.
// stderr_vs_best (optional): standard error of values[k] - values[best] (paired over deals).
int pimc_card_values(const Round& r, const SearchConfig& cfg, Rng& rng, DDSolver& dd, int* cards, double* values,
                     double* stderr_vs_best = nullptr);

// Same, but for the actual deal in r.hands (hindsight: all cards known).
int deal_card_values(const Round& r, const SearchConfig& cfg, Rng& rng, DDSolver& dd, int* cards, double* values);

// Mean final slate difference (own team minus opponents, multiplier applied, Weis and
// Stöck included) for declaring each trump mode 0..5 (out[0..5]) and for pushing (out[6],
// only meaningful if pushing is allowed). The partner's choice after a push is taken
// from `partner_choice` (the heuristic if null).
void pimc_trump_values(const Round& r, const SearchConfig& cfg, Rng& rng, DDSolver& dd, double out[7],
                       Agent* partner_choice = nullptr);

// Value (team 0 minus team 1, raw points of the remaining play) of a fully known round:
// rollout policy, then exact. `r` is advanced by the rollout.
int evaluate_deal(Round& r, const SearchConfig& cfg, Rng& rng, DDSolver& dd);

}  // namespace jass
