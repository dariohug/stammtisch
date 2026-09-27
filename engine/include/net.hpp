// Tiny MLP inference (Linear + ReLU), weights exported by stammtisch/train.py.
#pragma once
#include <memory>
#include <string>
#include <vector>

namespace jass {

class Mlp {
public:
    // Loads a 'JNN1' file; returns nullptr and sets *err on failure. Models are cached
    // per path and shared between threads (forward() is thread-safe).
    static std::shared_ptr<const Mlp> load(const std::string& path, std::string* err = nullptr);

    int in_size() const { return layers_.empty() ? 0 : layers_.front().in; }
    int out_size() const { return layers_.empty() ? 0 : layers_.back().out; }
    void forward(const float* x, float* out) const;

private:
    struct Layer {
        int in, out;
        std::vector<float> wt;  // transposed: [in][out], so sparse inputs skip whole rows
        std::vector<float> b;
    };
    std::vector<Layer> layers_;
};

// Directory with card_policy.bin / trump_policy.bin: $STAMMTISCH_MODELS or ./models.
std::string models_dir();

}  // namespace jass
