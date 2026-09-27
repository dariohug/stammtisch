#include "features.hpp"

#include <cstring>

#include "pimc.hpp"

namespace jass {

namespace {
void put_cards(float* out, Cards c) {
    for (; c; c &= c - 1) out[lowest(c)] = 1.0f;
}
}  // namespace

void card_features(const Round& r, float* out) {
    using namespace feat;
    std::memset(out, 0, sizeof(float) * kSize);
    const int me = r.to_play;
    put_cards(out + kHand, r.hands[me]);
    const int np = r.n_played();
    for (int k = 0; k < np; ++k) out[kPlayed + rel_seat(me, r.seat_of(k)) * 36 + r.history[k]] = 1.0f;
    for (int i = 0; i < r.n_in_trick; ++i) out[kTrick + i * 36 + r.trick[i]] = 1.0f;
    out[kTrump + r.trump] = 1.0f;
    if (r.declarer >= 0) out[kDeclarer + rel_seat(me, r.declarer)] = 1.0f;
    out[kPushed] = r.pushed ? 1.0f : 0.0f;
    const InfoSet is = info_set(r, me);
    for (int s = 0; s < 4; ++s) {
        if (s == me) continue;
        for (int su = 0; su < 4; ++su) {
            Cards possible = is.allowed[s] & kSuitMask[su];
            if (r.trump == su) possible &= ~bit(su * 9 + kJ);  // "no trump but maybe the Puur" counts as void
            if (!possible) out[kVoid + rel_seat(me, s) * 4 + su] = 1.0f;
        }
    }
    out[kLeader + rel_seat(me, r.trick_first)] = 1.0f;
    if (r.n_tricks >= 1)
        for (int s = 0; s < 4; ++s) put_cards(out + kShown + rel_seat(me, s) * 36, r.shown[s] & ~r.played);
    out[kScore] = r.points[team_of(me)] / 157.0f;
    out[kScore + 1] = r.points[1 - team_of(me)] / 157.0f;
    out[kScore + 2] = r.n_tricks / 9.0f;
}

void trump_features(Cards hand, bool pushed, float* out) {
    std::memset(out, 0, sizeof(float) * kTrumpFeatures);
    put_cards(out, hand);
    out[36] = pushed ? 1.0f : 0.0f;
}

}  // namespace jass
