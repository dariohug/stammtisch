// Network inputs, computed from what the acting player can see. Shared by training
// (via the C API) and play, so both always agree.
//
// Seats are relative to the acting player in play order: 0 = me, 1 = next (right-hand
// opponent), 2 = partner, 3 = previous opponent.
#pragma once
#include "jass.hpp"

namespace jass {

namespace feat {
constexpr int kHand = 0;          // 36  own hand
constexpr int kPlayed = 36;       // 4x36 cards played by each relative seat (incl. current trick)
constexpr int kTrick = 180;       // 3x36 current trick by position (lead, 2nd, 3rd)
constexpr int kTrump = 288;       // 6   trump mode
constexpr int kDeclarer = 294;    // 4   relative seat of the declarer
constexpr int kPushed = 298;      // 1
constexpr int kVoid = 299;        // 4x4 known voids per relative seat and suit
constexpr int kLeader = 315;      // 4   relative seat that led the current trick
constexpr int kShown = 319;       // 4x36 shown Weis cards not yet played
constexpr int kScore = 463;       // 3   own points/157, opponent points/157, tricks done/9
constexpr int kSize = 466;
}  // namespace feat

constexpr int kCardFeatures = feat::kSize;
constexpr int kTrumpFeatures = 37;  // hand + pushed
constexpr int kTrumpActions = 7;    // 6 modes + push

inline int rel_seat(int me, int s) { return (me - s + 4) & 3; }

// Features for r.to_play (trump must be set).
void card_features(const Round& r, float* out);
// Features for the trump chooser.
void trump_features(Cards hand, bool pushed, float* out);

}  // namespace jass
