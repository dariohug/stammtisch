// Built-in baseline agents. They see only what a real player sees
// (own hand + public history), never the other hands.
#pragma once
#include <string>

#include "jass.hpp"

namespace jass {

enum class AgentKind { Random, Heuristic };

bool parse_agent(const std::string& name, AgentKind& out);
const char* agent_name(AgentKind k);

// Trump choice for the seat returned by r.trump_chooser(). May return kPush
// only when !r.pushed.
int choose_trump(AgentKind k, const Round& r, Rng& rng);
// Card choice for r.to_play; always legal.
int choose_card(AgentKind k, const Round& r, Rng& rng);

// Classic DL4G hand-strength score for a trump mode (0..5).
int trump_score(Cards hand, int mode);

}  // namespace jass
