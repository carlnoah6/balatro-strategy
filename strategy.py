"""Build archetype tracking and strategic decision-making.

Tracks what build the player is pursuing (flush, pairs, straight, etc.)
and provides strategic context for all decisions.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from .scoring import Card, Joker, HandLevel, ScoreBreakdown, find_best_hands, calculate_score


class Archetype(Enum):
    """Build archetypes in Balatro."""
    UNDECIDED = "undecided"
    FLUSH = "flush"
    PAIRS = "pairs"          # Two Pair / Full House focus
    STRAIGHT = "straight"
    FOUR_KIND = "four_kind"  # Four of a Kind focus
    HIGH_MULT = "high_mult"  # xMult stacking (Glass, Polychrome)
    CHIP_STACK = "chip_stack"  # Raw chip stacking (Steel, Bonus)


# Jokers that strongly signal an archetype
ARCHETYPE_JOKERS = {
    Archetype.FLUSH: {
        "Splash", "Flower Pot", "Smeared Joker", "Bloodstone",
        "Arrowhead", "Onyx Agate", "Rough Gem",
    },
    Archetype.PAIRS: {
        "Mime", "Dusk", "Seltzer", "Sock and Buskin",
        "Hanging Chad", "Hack", "Jolly Joker", "Zany Joker",
        "Mad Joker", "Crazy Joker", "Sly Joker",
    },
    Archetype.STRAIGHT: {
        "Shortcut", "Four Fingers", "Run", "Wee Joker",
        "Fibonacci", "Even Steven", "Odd Todd",
    },
    Archetype.FOUR_KIND: {
        "The Duo", "The Trio", "The Family", "The Order", "The Tribe",
    },
    Archetype.HIGH_MULT: {
        "Obelisk", "Abstract Joker", "Misprint", "Ride the Bus",
        "Green Joker", "Red Card", "Hologram",
    },
}

# Hand types that signal an archetype
ARCHETYPE_HANDS = {
    Archetype.FLUSH: {"Flush", "Straight Flush", "Flush Five", "Flush House"},
    Archetype.PAIRS: {"Pair", "Two Pair", "Full House", "Flush House"},
    Archetype.STRAIGHT: {"Straight", "Straight Flush"},
    Archetype.FOUR_KIND: {"Four of a Kind", "Five of a Kind", "Flush Five"},
}


@dataclass
class ArchetypeTracker:
    """Tracks build archetype signals across the game."""
    scores: dict[str, float] = field(default_factory=lambda: {a.value: 0.0 for a in Archetype})
    committed: Optional[Archetype] = None
    commit_ante: int = 0
    hand_history: list[str] = field(default_factory=list)

    @property
    def current(self) -> Archetype:
        if self.committed:
            return self.committed
        if not any(v > 0 for v in self.scores.values()):
            return Archetype.UNDECIDED
        best = max(self.scores, key=self.scores.get)
        return Archetype(best)

    def signal_joker(self, joker_name: str, weight: float = 2.0):
        """Record a joker acquisition signal."""
        for arch, joker_set in ARCHETYPE_JOKERS.items():
            if joker_name in joker_set:
                self.scores[arch.value] += weight

    def signal_hand(self, hand_type: str, weight: float = 1.0):
        """Record a hand play signal."""
        self.hand_history.append(hand_type)
        for arch, hand_set in ARCHETYPE_HANDS.items():
            if hand_type in hand_set:
                self.scores[arch.value] += weight

    def signal_planet(self, hand_type: str, weight: float = 3.0):
        """Record a planet card usage — strong archetype signal."""
        for arch, hand_set in ARCHETYPE_HANDS.items():
            if hand_type in hand_set:
                self.scores[arch.value] += weight

    def try_commit(self, ante: int, threshold: float = 5.0) -> bool:
        """Try to commit to an archetype if signals are strong enough.

        Should be called around ante 2-3. Once committed, the archetype
        is locked and guides all future decisions.
        """
        if self.committed:
            return True
        best_arch = self.current
        if best_arch == Archetype.UNDECIDED:
            return False
        if self.scores[best_arch.value] >= threshold:
            self.committed = best_arch
            self.commit_ante = ante
            return True
        # Auto-commit by ante 3 to the strongest signal
        if ante >= 3 and best_arch != Archetype.UNDECIDED:
            self.committed = best_arch
            self.commit_ante = ante
            return True
        return False

    def archetype_summary(self) -> str:
        """Human-readable archetype status."""
        cur = self.current
        if self.committed:
            return f"Committed: {cur.value} (since ante {self.commit_ante})"
        top3 = sorted(self.scores.items(), key=lambda x: -x[1])[:3]
        signals = ", ".join(f"{k}={v:.1f}" for k, v in top3 if v > 0)
        return f"Exploring: {cur.value} | signals: {signals or 'none'}"


# ============================================================
# Strategic Decision Context
# ============================================================

@dataclass
class GameContext:
    """Full strategic context for decision-making."""
    ante: int = 1
    round_num: int = 0  # small=0, big=1, boss=2
    hands_left: int = 4
    discards_left: int = 3
    blind_chips: float = 0
    current_chips: float = 0
    dollars: int = 0
    hand_cards: list[Card] = field(default_factory=list)
    jokers: list[Joker] = field(default_factory=list)
    joker_slots: int = 5
    consumables: list[dict] = field(default_factory=list)
    consumable_slots: int = 2
    hand_levels: HandLevel = field(default_factory=HandLevel)
    archetype: ArchetypeTracker = field(default_factory=ArchetypeTracker)
    shop_items: list[dict] = field(default_factory=list)
    blind_info: dict = field(default_factory=dict)

    @property
    def chips_needed(self) -> float:
        return max(0, self.blind_chips - self.current_chips)

    @property
    def joker_space(self) -> int:
        return max(0, self.joker_slots - len(self.jokers))

    @property
    def consumable_space(self) -> int:
        return max(0, self.consumable_slots - len(self.consumables))

    @property
    def interest_money(self) -> int:
        """Money earning interest (capped at $25)."""
        return min(self.dollars // 5, 5)

    @property
    def above_interest_threshold(self) -> bool:
        """Whether we're at or above the max interest threshold."""
        return self.dollars >= 25

    @classmethod
    def from_state(cls, state: dict, archetype: ArchetypeTracker | None = None,
                   hand_levels: HandLevel | None = None) -> "GameContext":
        """Build context from raw game state dict."""
        hand = state.get("hand_cards", [])
        if isinstance(hand, dict):
            hand = list(hand.values()) if hand else []
        cards = [Card.from_state(c, i) for i, c in enumerate(hand)]

        jokers = state.get("jokers", [])
        if isinstance(jokers, dict):
            jokers = list(jokers.values()) if jokers else []
        joker_objs = [Joker.from_state(j) for j in jokers]

        shop = state.get("shop_items", [])
        if isinstance(shop, dict):
            shop = list(shop.values()) if shop else []

        # Build hand_levels from game state if available
        game_hl = state.get("hand_levels", {})
        if game_hl and isinstance(game_hl, dict) and any(isinstance(v, dict) for v in game_hl.values()):
            effective_hl = HandLevel.from_game_state(game_hl)
            # Merge with engine's tracked levels (engine may have more recent planet usage)
            if hand_levels:
                for ht, lvl in hand_levels.levels.items():
                    if lvl > effective_hl.levels.get(ht, 1):
                        effective_hl.levels[ht] = lvl
        else:
            effective_hl = hand_levels or HandLevel()

        return cls(
            ante=state.get("ante", 1),
            hands_left=state.get("hands_left", 4),
            discards_left=state.get("discards_left", 3),
            blind_chips=state.get("blind_chips", 0),
            current_chips=state.get("chips", 0),
            dollars=state.get("dollars", 0),
            hand_cards=cards,
            jokers=joker_objs,
            joker_slots=state.get("joker_slots", 5),
            consumables=state.get("consumables", []),
            consumable_slots=state.get("consumable_slots", 2),
            hand_levels=effective_hl,
            archetype=archetype or ArchetypeTracker(),
            shop_items=shop,
            blind_info=state.get("blind_info", {}),
        )


def build_context(state: dict, archetype: ArchetypeTracker | None = None,
                  hand_levels: HandLevel | None = None) -> GameContext:
    """Convenience wrapper for GameContext.from_state."""
    return GameContext.from_state(state, archetype, hand_levels)


# ============================================================
# Hand Strategy
# ============================================================

def should_discard(ctx: GameContext) -> tuple[bool, list[int], str]:
    """Decide whether to discard and which cards.

    Returns (should_discard, card_indices_to_discard, reasoning).
    """
    if ctx.discards_left <= 0:
        return (False, [], "No discards remaining")
    if not ctx.hand_cards:
        return (False, [], "No cards in hand")

    best_hands = find_best_hands(ctx.hand_cards, ctx.jokers, ctx.hand_levels, top_n=1)
    if not best_hands:
        return (False, [], "Cannot evaluate hand")

    best = best_hands[0]
    chips_needed = ctx.chips_needed

    # If we can already clear the blind, just play
    if best.final_score >= chips_needed and chips_needed > 0:
        return (False, [], f"Best hand ({best.hand_type}) scores {best.final_score:.0f} >= {chips_needed:.0f} needed")

    # If this is our last hand, must play
    if ctx.hands_left <= 1:
        return (False, [], "Last hand — must play")

    # Calculate discard value: what cards are NOT in the best hand?
    best_indices = set(best.all_cards)
    non_scoring = [i for i in range(len(ctx.hand_cards)) if i not in best_indices]

    # Archetype-aware discard: keep cards that fit our build
    arch = ctx.archetype.current
    discard_candidates = []

    for i in non_scoring:
        card = ctx.hand_cards[i]
        keep = False

        if arch == Archetype.FLUSH:
            # Keep cards matching the dominant suit
            suit_counts = Counter(c.suit for c in ctx.hand_cards)
            dominant_suit = suit_counts.most_common(1)[0][0]
            if card.suit == dominant_suit:
                keep = True

        elif arch == Archetype.PAIRS:
            # Keep cards with matching ranks
            rank_counts = Counter(c.rank for c in ctx.hand_cards)
            if rank_counts[card.rank] >= 2:
                keep = True

        elif arch == Archetype.STRAIGHT:
            # Keep cards that could form a straight
            ranks = sorted(set(c.rank_num for c in ctx.hand_cards))
            if _contributes_to_straight(card.rank_num, ranks):
                keep = True

        if not keep:
            discard_candidates.append(i)

    if not discard_candidates:
        # Nothing obvious to discard — check if hand is weak enough to warrant it
        if best.hand_rank <= 3 and ctx.hands_left > 1:
            # Weak hand (High Card / Pair / Two Pair), discard non-scoring cards aggressively
            discard_candidates = non_scoring[:min(5, ctx.discards_left)]
        elif best.final_score < chips_needed * 0.7 and ctx.hands_left > 1:
            # Score too low for target — discard non-scoring to try for better
            discard_candidates = non_scoring[:min(3, ctx.discards_left)]
        else:
            return (False, [], f"Hand is decent ({best.hand_type}), no clear discards")

    # Limit to available discards
    to_discard = discard_candidates[:ctx.discards_left]

    if not to_discard:
        return (False, [], "No cards worth discarding")

    reason = f"Discard {len(to_discard)} cards to improve {best.hand_type} (rank {best.hand_rank})"
    if arch != Archetype.UNDECIDED:
        reason += f" [build: {arch.value}]"

    return (True, to_discard, reason)


def choose_play(ctx: GameContext) -> tuple[list[int], str]:
    """Choose which cards to play.

    Returns (card_indices, reasoning).
    """
    if not ctx.hand_cards:
        return ([], "No cards")

    best_hands = find_best_hands(ctx.hand_cards, ctx.jokers, ctx.hand_levels, top_n=3)
    if not best_hands:
        return (list(range(min(5, len(ctx.hand_cards)))), "Fallback: play first 5")

    best = best_hands[0]
    chips_needed = ctx.chips_needed

    # If best hand clears the blind, play it
    if best.final_score >= chips_needed:
        reason = (f"Play {best.hand_type} for {best.final_score:.0f} "
                  f"(need {chips_needed:.0f}, overkill {best.final_score/max(1,chips_needed):.1f}x)")
    else:
        reason = (f"Play {best.hand_type} for {best.final_score:.0f} "
                  f"(need {chips_needed:.0f}, {best.final_score/max(1,chips_needed)*100:.0f}% of target)")

    return (best.all_cards, reason)


def _contributes_to_straight(rank: int, all_ranks: list[int]) -> bool:
    """Check if a rank contributes to a potential straight."""
    for base in range(max(1, rank - 4), rank + 1):
        window = set(range(base, base + 5))
        overlap = window & set(all_ranks)
        if rank in window and len(overlap) >= 3:
            return True
    # Ace-low
    if rank == 14:
        low_window = {14, 2, 3, 4, 5}
        if len(low_window & set(all_ranks)) >= 3:
            return True
    return False


# ============================================================
# Shop Strategy
# ============================================================

def evaluate_shop_item(item: dict, ctx: GameContext) -> tuple[float, str]:
    """Score a shop item from 0-10 based on strategic value.

    Returns (score, reasoning).
    """
    name = item.get("name", "")
    cost = item.get("cost", 0)
    item_type = item.get("type", "")

    if cost > ctx.dollars:
        return (0.0, "Can't afford")

    score = 5.0  # baseline
    reasons = []

    # Economy check: buying shouldn't drop below interest threshold
    money_after = ctx.dollars - cost
    interest_loss = max(0, ctx.interest_money - min(money_after // 5, 5))
    if interest_loss > 0 and ctx.ante >= 2:
        score -= interest_loss * 1.5
        reasons.append(f"loses ${interest_loss} interest")

    # Joker slot check
    if item_type == "Joker":
        if ctx.joker_space <= 0:
            return (0.0, "No joker slots")

        # Jokers are the core scaling mechanic — always valuable early
        if ctx.ante <= 3 and len([j for j in (ctx.jokers or []) if True]) < 3:
            score += 2.0
            reasons.append("early game, need jokers")

        # Archetype synergy
        arch = ctx.archetype.current
        for a, joker_set in ARCHETYPE_JOKERS.items():
            if name in joker_set:
                if a == arch:
                    score += 3.0
                    reasons.append(f"synergy with {arch.value} build")
                elif arch == Archetype.UNDECIDED:
                    score += 2.0
                    reasons.append(f"signals {a.value}")
                else:
                    score -= 1.0
                    reasons.append(f"off-archetype ({a.value})")
                break

    elif item_type == "Planet":
        if ctx.consumable_space <= 0:
            return (0.0, "No consumable slots")
        # Planet cards are good if they match our archetype
        # (planet name maps to hand type, but we'd need a lookup)
        score += 1.0
        reasons.append("planet card")

    elif item_type == "Tarot":
        if ctx.consumable_space <= 0:
            return (0.0, "No consumable slots")
        score += 0.5
        reasons.append("tarot card")

    elif item_type == "Voucher":
        score += 0.5
        reasons.append("voucher")

    # Early game: prioritize economy
    if ctx.ante <= 2 and cost > 4:
        score -= 1.0
        reasons.append("expensive for early game")

    # Late game: prioritize power
    if ctx.ante >= 5:
        score += 1.0
        reasons.append("late game — power matters more")

    return (max(0.0, min(10.0, score)), "; ".join(reasons) if reasons else "baseline")


def shop_decisions(ctx: GameContext) -> list[tuple[int, float, str]]:
    """Rank shop items by purchase priority.

    Returns list of (item_index, score, reasoning) sorted by score descending.
    """
    results = []
    for i, item in enumerate(ctx.shop_items):
        score, reason = evaluate_shop_item(item, ctx)
        results.append((i, score, reason))
    results.sort(key=lambda x: -x[1])
    return results
