#include "net.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>

namespace jass {

std::string models_dir() {
    const char* env = std::getenv("STAMMTISCH_MODELS");
    return env && *env ? env : "models";
}

std::shared_ptr<const Mlp> Mlp::load(const std::string& path, std::string* err) {
    static std::mutex mu;
    static std::map<std::string, std::shared_ptr<const Mlp>> cache;
    std::lock_guard<std::mutex> lock(mu);
    if (auto it = cache.find(path); it != cache.end()) return it->second;

    auto fail = [&](const std::string& msg) -> std::shared_ptr<const Mlp> {
        if (err) *err = path + ": " + msg;
        return nullptr;
    };
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return fail("cannot open (train it with `python -m stammtisch train`)");
    char magic[4];
    int32_t n = 0;
    auto m = std::make_shared<Mlp>();
    bool ok = std::fread(magic, 1, 4, f) == 4 && std::memcmp(magic, "JNN1", 4) == 0 &&
              std::fread(&n, 4, 1, f) == 1 && n > 0 && n < 64;
    for (int l = 0; ok && l < n; ++l) {
        int32_t dims[2];
        ok = std::fread(dims, 4, 2, f) == 2 && dims[0] > 0 && dims[1] > 0;
        if (!ok) break;
        Layer L{dims[0], dims[1], {}, {}};
        std::vector<float> w(static_cast<size_t>(L.in) * L.out);
        L.b.resize(static_cast<size_t>(L.out));
        ok = std::fread(w.data(), 4, w.size(), f) == w.size() && std::fread(L.b.data(), 4, L.b.size(), f) == L.b.size();
        L.wt.resize(w.size());
        for (int o = 0; o < L.out; ++o)
            for (int i = 0; i < L.in; ++i) L.wt[static_cast<size_t>(i) * L.out + o] = w[static_cast<size_t>(o) * L.in + i];
        if (!m->layers_.empty() && m->layers_.back().out != L.in) ok = false;
        m->layers_.push_back(std::move(L));
    }
    std::fclose(f);
    if (!ok) return fail("corrupt model file");
    cache[path] = m;
    return m;
}

void Mlp::forward(const float* x, float* out) const {
    thread_local std::vector<float> a, b;
    const float* cur = x;
    for (size_t l = 0; l < layers_.size(); ++l) {
        const Layer& L = layers_[l];
        std::vector<float>& y = (l % 2 == 0) ? a : b;
        y.assign(L.b.begin(), L.b.end());
        float* yp = y.data();
        for (int i = 0; i < L.in; ++i) {
            const float xi = cur[i];
            if (xi == 0.0f) continue;  // inputs and ReLU outputs are mostly zero
            const float* w = L.wt.data() + static_cast<size_t>(i) * L.out;
            for (int o = 0; o < L.out; ++o) yp[o] += xi * w[o];
        }
        if (l + 1 < layers_.size())
            for (int o = 0; o < L.out; ++o) yp[o] = yp[o] > 0.0f ? yp[o] : 0.0f;
        cur = yp;
    }
    std::memcpy(out, cur, sizeof(float) * static_cast<size_t>(out_size()));
}

}  // namespace jass
