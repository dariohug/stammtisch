#include "agents.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <vector>

#include "dd.hpp"
#include "features.hpp"
#include "net.hpp"
#include "pimc.hpp"

namespace jass {

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

}  // namespace

int heuristic_card(const Round& r) {
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
    return pick_by_value(legal, trump, false);
}

int heuristic_trump(const Round& r) {
    const Cards hand = r.hands[r.trump_chooser()];
    int best = 0, best_score = -1;
    for (int m = 0; m < 6; ++m) {
        int s = trump_score(hand, m);
        if (s > best_score) { best = m; best_score = s; }
    }
    if (!r.pushed && best_score < kPushThreshold) return kPush;
    return best;
}

namespace {

class RandomAgent : public Agent {
public:
    int choose_trump(const Round& r, Rng& rng) override {
        const int t = static_cast<int>(rng.below(r.pushed ? 6 : 7));
        return t == 6 ? kPush : t;
    }
    int choose_card(const Round& r, Rng& rng) override { return rng.pick(r.legal()); }
    std::string name() const override { return "random"; }
};

class HeuristicAgent : public Agent {
public:
    int choose_trump(const Round& r, Rng&) override { return heuristic_trump(r); }
    int choose_card(const Round& r, Rng&) override { return heuristic_card(r); }
    std::string name() const override { return "heuristic"; }
};

class NetAgent : public Agent {
public:
    NetAgent(std::shared_ptr<const Mlp> card, std::shared_ptr<const Mlp> trump, bool sample)
        : card_(std::move(card)), trump_(std::move(trump)), sample_(sample) {}

    int choose_trump(const Round& r, Rng& rng) override {
        float x[kTrumpFeatures], out[kTrumpActions];
        trump_features(r.hands[r.trump_chooser()], r.pushed, x);
        trump_->forward(x, out);
        const int n = r.pushed ? 6 : 7;
        const int a = pick(out, n, [](int) { return true; }, rng);
        return a == 6 ? kPush : a;
    }

    int choose_card(const Round& r, Rng& rng) override {
        const Cards legal = r.legal();
        if (popcount(legal) == 1) return lowest(legal);
        float x[kCardFeatures], out[36];
        card_features(r, x);
        card_->forward(x, out);
        return pick(out, 36, [&](int c) { return (legal >> c) & 1; }, rng);
    }

    std::string name() const override { return label_ + (sample_ ? ":sample" : ""); }
    std::string label_ = "net";

private:
    template <typename Ok>
    int pick(const float* logits, int n, Ok ok, Rng& rng) const {
        int best = -1;
        for (int i = 0; i < n; ++i)
            if (ok(i) && (best < 0 || logits[i] > logits[best])) best = i;
        if (!sample_) return best;
        double p[36], total = 0;
        for (int i = 0; i < n; ++i) total += p[i] = ok(i) ? std::exp(static_cast<double>(logits[i] - logits[best])) : 0.0;
        double u = (rng.next() >> 11) * (1.0 / 9007199254740992.0) * total;
        for (int i = 0; i < n; ++i)
            if ((u -= p[i]) <= 0 && p[i] > 0) return i;
        return best;
    }

    std::shared_ptr<const Mlp> card_, trump_;
    bool sample_;
};

// variant "" = imitation of humans (card_policy.bin), "strong" = distilled from search
// (card_policy_strong.bin).
std::unique_ptr<Agent> make_net(bool sample, std::string* err, const std::string& variant = "") {
    auto card = Mlp::load(models_dir() + (variant.empty() ? "/card_policy.bin" : "/card_policy_" + variant + ".bin"), err);
    if (!card) return nullptr;
    auto trump = Mlp::load(models_dir() + "/trump_policy.bin", err);
    if (!trump) return nullptr;
    if (card->in_size() != kCardFeatures || card->out_size() != 36 || trump->in_size() != kTrumpFeatures ||
        trump->out_size() != kTrumpActions) {
        if (err) *err = "model shapes do not match the feature layout - retrain";
        return nullptr;
    }
    auto agent = std::make_unique<NetAgent>(std::move(card), std::move(trump), sample);
    if (!variant.empty()) agent->label_ = "net:" + variant;
    return agent;
}

// Search agent: PIMC over sampled deals; each sample is played out by the rollout policy
// until `exact_left` tricks remain and then solved exactly.
class PimcAgent : public Agent {
public:
    PimcAgent(const SearchConfig& cfg, int trump_samples, std::unique_ptr<Agent> rollout)
        : cfg_(cfg), trump_samples_(trump_samples), rollout_(std::move(rollout)), dd_(21) {
        cfg_.rollout = rollout_.get();
    }

    int choose_trump(const Round& r, Rng& rng) override {
        if (trump_samples_ <= 0) return rollout_->choose_trump(r, rng);
        SearchConfig tc = cfg_;
        tc.samples = trump_samples_;
        double v[7];
        pimc_trump_values(r, tc, rng, dd_, v, rollout_.get());
        int best = 0;
        for (int m = 1; m < 6; ++m)
            if (v[m] > v[best]) best = m;
        if (!r.pushed && v[6] > v[best]) return kPush;
        return best;
    }

    int choose_card(const Round& r, Rng& rng) override {
        const Cards legal = r.legal();
        if (popcount(legal) == 1) return lowest(legal);
        int cards[9];
        double values[9];
        const int n = pimc_card_values(r, cfg_, rng, dd_, cards, values);
        int best = 0;
        for (int i = 1; i < n; ++i)
            if (values[i] > values[best] + 1e-9) best = i;
        return cards[best];
    }

    std::string name() const override {
        char b[32];
        std::snprintf(b, sizeof b, "%g", cfg_.belief);
        return "pimc:n=" + std::to_string(cfg_.samples) + ",t=" + std::to_string(trump_samples_) +
               ",k=" + std::to_string(cfg_.exact_left) + ",b=" + b + ",roll=" + rollout_->name();
    }

private:
    SearchConfig cfg_;
    int trump_samples_;
    std::unique_ptr<Agent> rollout_;
    DDSolver dd_;
};

}  // namespace

std::unique_ptr<Agent> make_agent(const std::string& spec, std::string* err) {
    const auto colon = spec.find(':');
    const std::string kind = spec.substr(0, colon);
    const std::string args = colon == std::string::npos ? "" : spec.substr(colon + 1);
    if (kind == "random" && args.empty()) return std::make_unique<RandomAgent>();
    if (kind == "heuristic" && args.empty()) return std::make_unique<HeuristicAgent>();
    if (kind == "net") {  // net[:strong][,sample]
        bool sample = false;
        std::string variant;
        std::stringstream ss(args);
        for (std::string tok; std::getline(ss, tok, ',');) {
            if (tok == "sample") sample = true;
            else if (tok == "strong") variant = "strong";
            else if (!tok.empty()) { if (err) *err = "unknown net option '" + tok + "'"; return nullptr; }
        }
        return make_net(sample, err, variant);
    }
    if (kind == "pimc") {
        // pimc[:n=32,t=0,k=5,b=0.5,c=4,roll=net|heuristic|strong]   (a bare number is n)
        int n = 32, t = 0, k = 5, cands = 4;  // t=0: trump by the policy (A/B: as strong, far cheaper)
        double belief = -1;  // default: 0.5 when the behaviour models are available
        std::string roll = "auto";
        std::stringstream ss(args);
        for (std::string kv; std::getline(ss, kv, ',');) {
            if (kv.empty()) continue;
            const auto eq = kv.find('=');
            const std::string key = eq == std::string::npos ? "n" : kv.substr(0, eq);
            const std::string val = eq == std::string::npos ? kv : kv.substr(eq + 1);
            if (key == "n") n = std::max(1, std::atoi(val.c_str()));
            else if (key == "t") t = std::max(0, std::atoi(val.c_str()));
            else if (key == "k") k = std::min(9, std::max(0, std::atoi(val.c_str())));
            else if (key == "roll") roll = val;
            else if (key == "b") belief = std::max(0.0, std::atof(val.c_str()));
            else if (key == "c") cands = std::max(1, std::atoi(val.c_str()));
            else { if (err) *err = "unknown pimc option '" + key + "'"; return nullptr; }
        }
        std::unique_ptr<Agent> rollout;
        if (roll == "net" || roll == "auto" || roll == "strong") {
            std::string net_err;
            rollout = make_net(false, &net_err, roll == "strong" ? "strong" : "");
            if (!rollout && roll != "auto") { if (err) *err = net_err; return nullptr; }
        }
        if (!rollout) rollout = std::make_unique<HeuristicAgent>();
        SearchConfig cfg;
        cfg.samples = n;
        cfg.exact_left = k;
        cfg.candidates = cands;
        auto card = Mlp::load(models_dir() + "/card_policy.bin");
        auto trump = Mlp::load(models_dir() + "/trump_policy.bin");
        if (card && trump) {  // cached for the whole process, so raw pointers stay valid
            cfg.card_model = card.get();
            cfg.trump_model = trump.get();
            cfg.belief = belief < 0 ? 0.5 : belief;
        } else {
            cfg.belief = 0;
        }
        return std::make_unique<PimcAgent>(cfg, t, std::move(rollout));
    }
    if (err) *err = "unknown agent spec '" + spec + "' (random | heuristic | net[:sample] | pimc[:n=,t=,k=,roll=])";
    return nullptr;
}

}  // namespace jass
