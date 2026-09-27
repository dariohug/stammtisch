#include "pimc.hpp"

#include <cmath>

#include "agents.hpp"
#include "features.hpp"
#include "net.hpp"

namespace jass {

InfoSet info_set(const Round& r, int me) {
    InfoSet is{};
    is.me = me;
    int remaining[4] = {9, 9, 9, 9};
    for (int s = 0; s < 4; ++s) is.allowed[s] = kAll;

    const int np = r.n_played();
    for (int k = 0; k < np; ++k) {
        const int seat = r.seat_of(k);
        remaining[seat]--;
        const int lead_card = r.history[k - k % 4];
        const int c = r.history[k];
        const int lead = suit_of(lead_card);
        if (k % 4 == 0 || suit_of(c) == lead) continue;
        if (r.trump < kObe && lead == r.trump) {
            // Did not follow trump: holds no trump except possibly the Puur.
            is.allowed[seat] &= ~(kSuitMask[lead] & ~bit(lead * 9 + kJ));
        } else if (!(r.trump < kObe && suit_of(c) == r.trump)) {
            is.allowed[seat] &= ~kSuitMask[lead];  // neither followed nor trumped: void
        }
    }
    Cards known = r.played | r.hands[me];
    for (int s = 0; s < 4; ++s) {
        is.fixed[s] = 0;
        if (s == me) is.fixed[s] = r.hands[me];
        else if (r.n_tricks >= 1) is.fixed[s] = r.shown[s] & ~r.played;
        known |= is.fixed[s];
    }
    is.pool = kAll & ~known;
    for (int s = 0; s < 4; ++s) is.need[s] = s == me ? 0 : remaining[s] - popcount(is.fixed[s]);
    return is;
}

bool sample_deal(const InfoSet& is, Rng& rng, Cards out[4]) {
    int cards[36], n = 0;
    for (Cards p = is.pool; p; p &= p - 1) cards[n++] = lowest(p);

    for (int attempt = 0; attempt < 64; ++attempt) {
        const bool relax = attempt == 63;
        // Most constrained cards first, random order among equals.
        int key[36];
        for (int i = 0; i < n; ++i) {
            int elig = 0;
            for (int s = 0; s < 4; ++s)
                if (is.need[s] > 0 && (is.allowed[s] & bit(cards[i]))) ++elig;
            key[i] = elig * 64 + static_cast<int>(rng.below(64));
        }
        for (int i = 1; i < n; ++i) {
            const int c = cards[i], k = key[i];
            int j = i - 1;
            while (j >= 0 && key[j] > k) { cards[j + 1] = cards[j]; key[j + 1] = key[j]; --j; }
            cards[j + 1] = c;
            key[j + 1] = k;
        }
        int need[4];
        for (int s = 0; s < 4; ++s) { need[s] = is.need[s]; out[s] = is.fixed[s]; }
        bool ok = true;
        for (int i = 0; i < n && ok; ++i) {
            const int c = cards[i];
            int total = 0;
            for (int s = 0; s < 4; ++s)
                if (need[s] > 0 && (relax || (is.allowed[s] & bit(c)))) total += need[s];
            if (!total) { ok = false; break; }
            int x = static_cast<int>(rng.below(static_cast<uint32_t>(total)));
            for (int s = 0; s < 4; ++s) {
                if (need[s] <= 0 || !(relax || (is.allowed[s] & bit(c)))) continue;
                if (x < need[s]) { out[s] |= bit(c); need[s]--; break; }
                x -= need[s];
            }
        }
        if (ok) return !relax;
    }
    return false;
}

int evaluate_deal(Round& r, const SearchConfig& cfg, Rng& rng, DDSolver& dd) {
    const int p0 = r.points[0], p1 = r.points[1];
    // Tricks only complete at trick boundaries, so the exact part starts at a trick start
    // (or wherever we already were, if inside the exact zone).
    while (9 - r.n_tricks > cfg.exact_left)
        r.play(cfg.rollout ? cfg.rollout->choose_card(r, rng) : heuristic_card(r));
    return (r.points[0] - p0) - (r.points[1] - p1) + dd.value(Position::from_round(r));
}

namespace {

double log_softmax_at(const float* logits, int n, int idx, Cards allowed) {
    float mx = -1e30f;
    for (int i = 0; i < n; ++i)
        if ((allowed >> i & 1) && logits[i] > mx) mx = logits[i];
    double sum = 0;
    for (int i = 0; i < n; ++i)
        if (allowed >> i & 1) sum += std::exp(static_cast<double>(logits[i] - mx));
    return static_cast<double>(logits[idx] - mx) - std::log(sum);
}

}  // namespace

double action_log_likelihood(const Round& r, int me, const Cards current[4], const SearchConfig& cfg) {
    // Rebuild the deal: current hands plus everything each seat has played.
    Cards initial[4];
    for (int s = 0; s < 4; ++s) initial[s] = current[s];
    const int np = r.n_played();
    for (int k = 0; k < np; ++k) initial[r.seat_of(k)] |= bit(r.history[k]);

    double ll = 0;
    const int fh = r.forehand();
    if (cfg.trump_model && r.trump >= 0) {
        float x[kTrumpFeatures], out[kTrumpActions];
        if (fh != me) {  // forehand's decision: declare or push
            trump_features(initial[fh], false, x);
            cfg.trump_model->forward(x, out);
            ll += log_softmax_at(out, 7, r.pushed ? 6 : r.trump, 0x7F);
        }
        if (r.pushed && partner(fh) != me) {
            trump_features(initial[partner(fh)], true, x);
            cfg.trump_model->forward(x, out);
            ll += log_softmax_at(out, 7, r.trump, 0x3F);
        }
    }
    if (cfg.card_model) {
        Round t;
        t.reset(initial, r.dealer, r.rules);
        if (r.pushed) t.choose_trump(kPush);
        t.choose_trump(r.trump);
        float x[kCardFeatures], out[36];
        for (int k = 0; k < np; ++k) {
            const int c = r.history[k];
            if (t.to_play != me) {
                const Cards legal = t.legal();
                if (popcount(legal) > 1) {
                    card_features(t, x);
                    cfg.card_model->forward(x, out);
                    ll += log_softmax_at(out, 36, c, legal);
                }
            }
            t.play(c);
        }
    }
    return ll;
}

void draw_deals(const Round& r, int me, const SearchConfig& cfg, Rng& rng, std::vector<std::array<Cards, 4>>& out) {
    const InfoSet is = info_set(r, me);
    out.clear();
    const bool weighted = cfg.belief > 0 && (cfg.card_model || cfg.trump_model) && r.n_played() + (r.pushed ? 1 : 0) > 0;
    if (!weighted) {
        for (int i = 0; i < cfg.samples; ++i) {
            std::array<Cards, 4> h;
            sample_deal(is, rng, h.data());
            out.push_back(h);
        }
        return;
    }
    const int m = cfg.samples * std::max(1, cfg.candidates);
    std::vector<std::array<Cards, 4>> cand(static_cast<size_t>(m));
    std::vector<double> logw(static_cast<size_t>(m));
    double mx = -1e300;
    for (int i = 0; i < m; ++i) {
        sample_deal(is, rng, cand[i].data());
        logw[i] = cfg.belief * action_log_likelihood(r, me, cand[i].data(), cfg);
        mx = std::max(mx, logw[i]);
    }
    double total = 0;
    for (auto& w : logw) total += (w = std::exp(w - mx));
    // Systematic resampling.
    const double step = total / cfg.samples;
    double u = (rng.next() >> 11) * (1.0 / 9007199254740992.0) * step, acc = 0;
    int j = 0;
    for (int i = 0; i < cfg.samples; ++i) {
        const double target = u + i * step;
        while (j < m - 1 && acc + logw[j] < target) acc += logw[j++];
        out.push_back(cand[j]);
    }
}

// Adds the value of each candidate card in one fully known deal.
static void add_deal_values(const Round& full, const SearchConfig& cfg, Rng& rng, DDSolver& dd, const int* cards,
                            int n, double* values) {
    const int sign = team_of(full.to_play) == 0 ? 1 : -1;
    if (9 - full.n_tricks <= cfg.exact_left) {
        int mc[9], mv[9];
        const int m = dd.move_values(Position::from_round(full), mc, mv);
        for (int j = 0; j < m; ++j)
            for (int k = 0; k < n; ++k)
                if (cards[k] == mc[j]) values[k] += mv[j];
        return;
    }
    for (int k = 0; k < n; ++k) {
        Round t = full;
        const int p0 = t.points[0], p1 = t.points[1];
        t.play(cards[k]);
        const int gained = (t.points[0] - p0) - (t.points[1] - p1);
        values[k] += sign * (gained + evaluate_deal(t, cfg, rng, dd));
    }
}

int pimc_card_values(const Round& r, const SearchConfig& cfg, Rng& rng, DDSolver& dd, int* cards, double* values,
                     double* stderr_vs_best) {
    int n = 0;
    for (Cards l = r.legal(); l; l &= l - 1) { cards[n] = lowest(l); values[n++] = 0; }
    if (n == 1) {
        if (stderr_vs_best) stderr_vs_best[0] = 0;
        return 1;
    }
    thread_local std::vector<std::array<Cards, 4>> deals;
    thread_local std::vector<std::array<double, 9>> per_sample;
    draw_deals(r, r.to_play, cfg, rng, deals);
    per_sample.assign(deals.size(), {});
    Round tmp = r;
    for (size_t i = 0; i < deals.size(); ++i) {
        for (int s = 0; s < 4; ++s) tmp.hands[s] = deals[i][s];
        double v[9] = {};
        add_deal_values(tmp, cfg, rng, dd, cards, n, v);
        for (int k = 0; k < n; ++k) { values[k] += v[k]; per_sample[i][k] = v[k]; }
    }
    const double S = static_cast<double>(deals.size());
    for (int k = 0; k < n; ++k) values[k] /= S;
    if (stderr_vs_best) {
        // Paired standard error of (value[k] - value[best]): all cards share the same deals.
        int best = 0;
        for (int k = 1; k < n; ++k)
            if (values[k] > values[best]) best = k;
        for (int k = 0; k < n; ++k) {
            double m = values[k] - values[best], ss = 0;
            for (const auto& ps : per_sample) {
                const double d = ps[k] - ps[best] - m;
                ss += d * d;
            }
            stderr_vs_best[k] = S > 1 ? std::sqrt(ss / (S - 1) / S) : 0.0;
        }
    }
    return n;
}

int deal_card_values(const Round& r, const SearchConfig& cfg, Rng& rng, DDSolver& dd, int* cards, double* values) {
    int n = 0;
    for (Cards l = r.legal(); l; l &= l - 1) { cards[n] = lowest(l); values[n++] = 0; }
    add_deal_values(r, cfg, rng, dd, cards, n, values);
    return n;
}

void pimc_trump_values(const Round& r, const SearchConfig& cfg, Rng& rng, DDSolver& dd, double out[7],
                       Agent* partner_choice) {
    const int me = r.trump_chooser();
    const int team = team_of(me);
    for (int m = 0; m < 7; ++m) out[m] = 0;
    // Before trump, nothing is public: the other 27 cards are dealt uniformly.
    InfoSet is{};
    is.me = me;
    is.pool = kAll & ~r.hands[me];
    for (int s = 0; s < 4; ++s) {
        is.allowed[s] = kAll;
        is.fixed[s] = s == me ? r.hands[me] : 0;
        is.need[s] = s == me ? 0 : 9;
    }
    std::vector<std::array<Cards, 4>> deals;
    if (r.pushed && cfg.belief > 0 && cfg.trump_model) {
        // The forehand pushed: hands without a clear trump call are more likely.
        const int m = cfg.samples * std::max(1, cfg.candidates);
        std::vector<std::array<Cards, 4>> cand(static_cast<size_t>(m));
        std::vector<double> w(static_cast<size_t>(m));
        double total = 0;
        for (int i = 0; i < m; ++i) {
            sample_deal(is, rng, cand[i].data());
            float x[kTrumpFeatures], o[kTrumpActions];
            trump_features(cand[i][r.forehand()], false, x);
            cfg.trump_model->forward(x, o);
            total += w[i] = std::exp(cfg.belief * log_softmax_at(o, 7, 6, 0x7F));
        }
        const double step = total / cfg.samples;
        double u = (rng.next() >> 11) * (1.0 / 9007199254740992.0) * step, acc = 0;
        int j = 0;
        for (int i = 0; i < cfg.samples; ++i) {
            while (j < m - 1 && acc + w[j] < u + i * step) acc += w[j++];
            deals.push_back(cand[j]);
        }
    } else {
        for (int i = 0; i < cfg.samples; ++i) {
            std::array<Cards, 4> h;
            sample_deal(is, rng, h.data());
            deals.push_back(h);
        }
    }
    for (int i = 0; i < cfg.samples; ++i) {
        Cards deal_h[4] = {deals[i][0], deals[i][1], deals[i][2], deals[i][3]};
        double v[6];
        for (int m = 0; m < 6; ++m) {
            Round t;
            t.reset(deal_h, r.dealer, r.rules);
            t.pushed = r.pushed;
            t.choose_trump(m);
            const int d = evaluate_deal(t, cfg, rng, dd);
            const int card_diff = team == 0 ? d : -d;
            v[m] = t.rules.mult[m] * (card_diff + t.bonus[team] - t.bonus[1 - team]);
            out[m] += v[m];
        }
        if (!r.pushed) {
            Round t;
            t.reset(deal_h, r.dealer, r.rules);
            t.choose_trump(kPush);
            int choice = partner_choice ? partner_choice->choose_trump(t, rng) : heuristic_trump(t);
            if (choice < 0 || choice > 5) choice = 0;
            out[6] += v[choice];
        }
    }
    for (int m = 0; m < 7; ++m) out[m] /= cfg.samples;
}

}  // namespace jass
