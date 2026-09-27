// jass-sim: multi-threaded Schieber self-play simulator.
//
// Emits one JSON object per line: periodic {"type":"heartbeat",...} records while
// running and a final {"type":"summary",...}. `stammtisch monitor` renders them live.
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <mutex>
#include <thread>
#include <vector>

#include "agents.hpp"
#include "jass.hpp"

using namespace jass;
using Clock = std::chrono::steady_clock;

namespace {

struct Options {
    uint64_t games = 1'000'000;  // deals (x2 rounds with --duplicate)
    double seconds = 0;          // if > 0: run for this long instead
    int threads = static_cast<int>(std::thread::hardware_concurrency());
    std::string a = "heuristic", b = "random";
    std::string rules_name = "swisslos";
    Rules rules = Rules::swisslos();
    bool duplicate = true;
    uint64_t seed = 1;
    double interval = 0.5;
    uint64_t batch = 256;  // deals grabbed per work item (set to 1 for slow agents)
    FILE* out = stdout;
    FILE* log = nullptr;  // --log: binary round records (see write_record)
};

// Per-thread counters, cache-line aligned; written by the worker, read by the reporter.
struct alignas(64) Stats {
    std::atomic<uint64_t> rounds{0}, deals{0}, cards{0};
    std::atomic<int64_t> diff_sum{0};     // score(A) - score(B), per deal
    std::atomic<uint64_t> diff_sq{0};
    std::atomic<uint64_t> points_a{0}, points_b{0};
    std::atomic<uint64_t> wins_a{0}, wins_b{0};  // per round, by points
    std::atomic<uint64_t> matches_a{0}, matches_b{0};
    std::atomic<uint64_t> trump[6]{};
    std::atomic<uint64_t> pushes{0};
};

std::atomic<uint64_t> g_next_deal{0};
std::mutex g_log_mu;

// 44 bytes per round: dealer, trump, pushed, agent A's team (0/1), 36 cards in play
// order, and 4 bytes padding. Read by stammtisch/train.py (distillation).
void write_record(FILE* f, const Round& r, int team_a) {
    int8_t rec[44] = {};
    rec[0] = static_cast<int8_t>(r.dealer);
    rec[1] = static_cast<int8_t>(r.trump);
    rec[2] = r.pushed ? 1 : 0;
    rec[3] = static_cast<int8_t>(team_a);
    for (int k = 0; k < 36; ++k) rec[4 + k] = static_cast<int8_t>(r.history[k]);
    std::lock_guard<std::mutex> lock(g_log_mu);
    std::fwrite(rec, 1, sizeof rec, f);
}
std::atomic<bool> g_stop{false};

void add(std::atomic<uint64_t>& a, uint64_t v) { a.store(a.load(std::memory_order_relaxed) + v, std::memory_order_relaxed); }

// Plays one round. seat_agent[p] is the agent of seat p; returns points of team 0 / 1.
void play_round(Round& r, Agent* const seat_agent[4], Rng& rng, Stats& st) {
    while (r.trump < 0) {
        const int t = seat_agent[r.trump_chooser()]->choose_trump(r, rng);
        if (t == kPush) add(st.pushes, 1);
        r.choose_trump(t);
    }
    add(st.trump[r.trump], 1);
    while (!r.done()) r.play(seat_agent[r.to_play]->choose_card(r, rng));
    add(st.cards, 36);
    add(st.rounds, 1);
}

void worker(const Options& o, int tid, Stats& st) {
    Rng rng(o.seed * 0x100000001B3ull + static_cast<uint64_t>(tid) * 7919 + 17);
    Round r;
    Cards hands[4];
    auto agent_a = make_agent(o.a), agent_b = make_agent(o.b);  // per thread: own search tables
    Agent* const ab[4] = {agent_a.get(), agent_b.get(), agent_a.get(), agent_b.get()};  // A on seats 0/2
    Agent* const ba[4] = {agent_b.get(), agent_a.get(), agent_b.get(), agent_a.get()};  // A on seats 1/3
    const uint64_t kBatch = o.batch;
    while (!g_stop.load(std::memory_order_relaxed)) {
        const uint64_t start = g_next_deal.fetch_add(kBatch, std::memory_order_relaxed);
        if (o.seconds <= 0 && start >= o.games) break;
        const uint64_t end = o.seconds > 0 ? start + kBatch : std::min(start + kBatch, o.games);
        for (uint64_t g = start; g < end; ++g) {
            // Each deal gets its own RNG stream for the cards, so runs are reproducible.
            Rng deal_rng(o.seed ^ (g * 0x9E3779B97F4A7C15ull));
            deal(deal_rng, hands);
            const int dealer = static_cast<int>(g & 3);
            int score_a = 0, score_b = 0;
            for (int pass = 0; pass < (o.duplicate ? 2 : 1); ++pass) {
                Agent* const* seats = pass == 0 ? ab : ba;
                r.reset(hands, dealer, o.rules);
                play_round(r, seats, rng, st);
                if (o.log) write_record(o.log, r, pass == 0 ? 0 : 1);
                const int ta = pass == 0 ? 0 : 1;  // team of agent A this pass
                const int sa = r.score(ta), sb = r.score(1 - ta);
                score_a += sa;
                score_b += sb;
                if (sa > sb) add(st.wins_a, 1); else if (sb > sa) add(st.wins_b, 1);
                if (r.tricks_won[ta] == 9) add(st.matches_a, 1);
                if (r.tricks_won[1 - ta] == 9) add(st.matches_b, 1);
            }
            const int64_t d = score_a - score_b;
            st.diff_sum.store(st.diff_sum.load(std::memory_order_relaxed) + d, std::memory_order_relaxed);
            add(st.diff_sq, static_cast<uint64_t>(d * d));
            add(st.points_a, static_cast<uint64_t>(score_a));
            add(st.points_b, static_cast<uint64_t>(score_b));
            add(st.deals, 1);
        }
    }
}

struct Totals {
    uint64_t rounds = 0, deals = 0, cards = 0, pa = 0, pb = 0, wa = 0, wb = 0, ma = 0, mb = 0, pushes = 0;
    uint64_t trump[6]{};
    int64_t diff = 0;
    uint64_t diff_sq = 0;
};

Totals collect(const std::vector<Stats>& stats, std::vector<uint64_t>* per_thread) {
    Totals t;
    for (const auto& s : stats) {
        auto ld = [](const std::atomic<uint64_t>& a) { return a.load(std::memory_order_relaxed); };
        t.rounds += ld(s.rounds); t.deals += ld(s.deals); t.cards += ld(s.cards);
        t.pa += ld(s.points_a); t.pb += ld(s.points_b);
        t.wa += ld(s.wins_a); t.wb += ld(s.wins_b);
        t.ma += ld(s.matches_a); t.mb += ld(s.matches_b);
        t.pushes += ld(s.pushes);
        for (int i = 0; i < 6; ++i) t.trump[i] += ld(s.trump[i]);
        t.diff += s.diff_sum.load(std::memory_order_relaxed);
        t.diff_sq += ld(s.diff_sq);
        if (per_thread) per_thread->push_back(ld(s.rounds));
    }
    return t;
}

void emit(const Options& o, const char* type, double elapsed, double rate, const Totals& t,
          const std::vector<uint64_t>& per_thread) {
    const double n = t.deals ? static_cast<double>(t.deals) : 1.0;
    const double mean = t.diff / n;
    const double var = std::max(0.0, t.diff_sq / n - mean * mean);
    const double ci95 = t.deals > 1 ? 1.96 * std::sqrt(var / n) : 0.0;
    const double per = o.duplicate ? 2.0 : 1.0;
    std::fprintf(o.out,
        "{\"type\":\"%s\",\"t\":%.3f,\"threads\":%d,\"agent_a\":\"%s\",\"agent_b\":\"%s\",\"rules\":\"%s\","
        "\"duplicate\":%s,\"target\":%llu,\"seconds\":%.1f,"
        "\"rounds\":%llu,\"deals\":%llu,\"cards\":%llu,\"rounds_per_s\":%.1f,"
        "\"mean_points_a\":%.3f,\"mean_points_b\":%.3f,\"diff_per_round\":%.3f,\"diff_ci95\":%.3f,"
        "\"wins_a\":%llu,\"wins_b\":%llu,\"matches_a\":%llu,\"matches_b\":%llu,\"pushes\":%llu,"
        "\"trump\":[%llu,%llu,%llu,%llu,%llu,%llu],\"per_thread_rounds\":[",
        type, elapsed, o.threads, o.a.c_str(), o.b.c_str(), o.rules_name.c_str(), o.duplicate ? "true" : "false",
        static_cast<unsigned long long>(o.seconds > 0 ? 0 : o.games), o.seconds,
        (unsigned long long)t.rounds, (unsigned long long)t.deals, (unsigned long long)t.cards, rate,
        t.pa / n / per, t.pb / n / per, mean / per, ci95 / per,
        (unsigned long long)t.wa, (unsigned long long)t.wb, (unsigned long long)t.ma, (unsigned long long)t.mb,
        (unsigned long long)t.pushes,
        (unsigned long long)t.trump[0], (unsigned long long)t.trump[1], (unsigned long long)t.trump[2],
        (unsigned long long)t.trump[3], (unsigned long long)t.trump[4], (unsigned long long)t.trump[5]);
    for (size_t i = 0; i < per_thread.size(); ++i)
        std::fprintf(o.out, "%s%llu", i ? "," : "", (unsigned long long)per_thread[i]);
    std::fprintf(o.out, "]}\n");
    std::fflush(o.out);
}

void usage() {
    std::puts(
        "usage: jass-sim [options]\n"
        "  --games N         deals to play (default 1000000)\n"
        "  --seconds S       run for S seconds instead of a fixed number of deals\n"
        "  --threads K       worker threads (default: all cores)\n"
        "  --a AGENT         agent A: random | heuristic | pimc[:N[:T]] (default heuristic)\n"
        "  --b AGENT         agent B (default random)\n"
        "  --rules R         swisslos (Weis, Stoeck, x1/x2/x3; default) | plain (card points only)\n"
        "  --no-duplicate    play each deal once instead of twice with swapped seats\n"
        "  --seed N          RNG seed (default 1)\n"
        "  --interval S      heartbeat interval in seconds (default 0.5)\n"
        "  --out FILE        write JSONL to FILE instead of stdout\n"
        "  --log FILE        append every round (44-byte binary records) for training");
}

}  // namespace

int main(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto val = [&]() -> const char* {
            if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", a.c_str()); std::exit(2); }
            return argv[++i];
        };
        if (a == "--games") o.games = std::strtoull(val(), nullptr, 10);
        else if (a == "--seconds") o.seconds = std::atof(val());
        else if (a == "--threads") o.threads = std::max(1, std::atoi(val()));
        else if (a == "--a" || a == "--b") {
            const char* v = val();
            std::string err;
            if (!make_agent(v, &err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 2; }
            (a == "--a" ? o.a : o.b) = v;
        } else if (a == "--rules") {
            o.rules_name = val();
            if (o.rules_name == "swisslos") o.rules = Rules::swisslos();
            else if (o.rules_name == "plain") o.rules = Rules::plain();
            else { std::fprintf(stderr, "unknown rules %s\n", o.rules_name.c_str()); return 2; }
        } else if (a == "--no-duplicate") o.duplicate = false;
        else if (a == "--seed") o.seed = std::strtoull(val(), nullptr, 10);
        else if (a == "--interval") o.interval = std::atof(val());
        else if (a == "--log") {
            o.log = std::fopen(val(), "ab");
            if (!o.log) { std::perror("--log"); return 2; }
        } else if (a == "--out") {
            o.out = std::fopen(val(), "w");
            if (!o.out) { std::perror("--out"); return 2; }
        } else if (a == "-h" || a == "--help") { usage(); return 0; }
        else { std::fprintf(stderr, "unknown option %s\n", a.c_str()); usage(); return 2; }
    }

    if (o.a.rfind("pimc", 0) == 0 || o.b.rfind("pimc", 0) == 0) o.batch = 1;
    std::vector<Stats> stats(static_cast<size_t>(o.threads));
    std::vector<std::thread> pool;
    const auto t0 = Clock::now();
    for (int t = 0; t < o.threads; ++t) pool.emplace_back(worker, std::cref(o), t, std::ref(stats[t]));

    auto elapsed = [&] { return std::chrono::duration<double>(Clock::now() - t0).count(); };
    uint64_t last_rounds = 0;
    double last_t = 0;
    std::atomic<int> running{o.threads};
    std::thread reporter([&] {
        double next_beat = o.interval;
        while (running.load() > 0) {
            // Short ticks so the --seconds deadline holds regardless of the heartbeat interval.
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
            if (running.load() == 0) break;
            if (o.seconds > 0 && elapsed() >= o.seconds) g_stop = true;
            if (elapsed() < next_beat) continue;
            next_beat += o.interval;
            std::vector<uint64_t> per;
            Totals t = collect(stats, &per);
            double now = elapsed();
            emit(o, "heartbeat", now, (t.rounds - last_rounds) / std::max(1e-9, now - last_t), t, per);
            last_rounds = t.rounds;
            last_t = now;
        }
    });
    for (auto& th : pool) th.join();
    running = 0;
    reporter.join();

    std::vector<uint64_t> per;
    Totals t = collect(stats, &per);
    const double secs = elapsed();
    emit(o, "summary", secs, t.rounds / secs, t, per);
    if (o.out != stdout) std::fclose(o.out);
    if (o.log) std::fclose(o.log);
    return 0;
}
