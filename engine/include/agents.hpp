// Agents. They see only what a real player sees (own hand, public history, shown
// Weis), never the other hands.
#pragma once
#include <memory>
#include <string>

#include "jass.hpp"

namespace jass {

class Agent {
public:
    virtual ~Agent() = default;
    // Trump for r.trump_chooser(); may return kPush only when !r.pushed.
    virtual int choose_trump(const Round& r, Rng& rng) = 0;
    // Card for r.to_play; always legal.
    virtual int choose_card(const Round& r, Rng& rng) = 0;
    virtual std::string name() const = 0;
};

// Specs:  random | heuristic | net[:strong][,sample] | pimc[:n=32,t=0,k=5,b=0.5,c=4,roll=net]
//   net          imitation policy (card + trump), argmax; "sample" samples from it;
//                "strong" uses the policy distilled from search (card_policy_strong.bin)
//   pimc         n sampled deals per card (belief-weighted with strength b, c candidates each),
//                rollouts by `roll` until k tricks remain, then exact; trump by the policy
//                (t=0) or by searching t deals per mode
// Returns nullptr and sets *err for an unknown spec.
std::unique_ptr<Agent> make_agent(const std::string& spec, std::string* err = nullptr);

// Classic DL4G hand-strength score for a trump mode (0..5).
int trump_score(Cards hand, int mode);
// Heuristic trump choice (push below a threshold).
int heuristic_trump(const Round& r);
// Heuristic card choice.
int heuristic_card(const Round& r);

}  // namespace jass
