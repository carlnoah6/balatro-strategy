"""Decision engine — orchestrates scoring, strategy, and LLM advisor.

This is the single entry point for ai-agent.py to make decisions.
It replaces the scattered decision logic in the original monolith.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

from .scoring import (
    Card, Joker, HandLevel, ScoreBreakdown,
    find_best_hands, calculate_score,
)
from .strategy import (
    Archetype, ArchetypeTracker, GameContext,
    should_discard, choose_play, shop_decisions, evaluate_shop_item,
    build_context, get_boss_counter, should_reroll,
    JokerTier, JOKER_TIERS, TIER_SCORE_BONUS,
)
from .llm_advisor import (
    advise_discard, advise_shop, advise_boss,
    get_llm_stats,
)

USE_LLM = os.environ.get("USE_LLM", "1") == "1"

# When to escalate to LLM (thresholds)
LLM_HAND_RANK_THRESHOLD = 2   # Ask LLM only for very weak hands (High Card, Pair)
LLM_SCORE_MARGIN = 0.5        # Ask LLM only if best hand < 50% of target


@dataclass
class Decision:
    """A decision with action, parameters, and reasoning."""
    action: str          # "play", "discard", "buy", "skip", "select_blind"
    params: dict = field(default_factory=dict)
    reasoning: str = ""
    source: str = "rule"  # "rule" or "llm"
    score_estimate: float = 0.0
    hand_type: str = ""   # e.g. "Flush", "Two Pair"


class DecisionEngine:
    """Stateful decision engine that persists across the game."""

    def __init__(self):
        self.archetype = ArchetypeTracker()
        self.hand_levels = HandLevel()
        self.game_count = 0
        self._last_ante = 0

    def new_game(self):
        """Reset for a new game run."""
        self.archetype = ArchetypeTracker()
        self.hand_levels = HandLevel()
        self.game_count += 1
        self._last_ante = 0

    def _build_context(self, state: dict) -> GameContext:
        """Build GameContext from raw game state."""
        ctx = build_context(state)
        ctx.archetype = self.archetype
        ctx.hand_levels = self.hand_levels
        return ctx

    def _should_use_llm(self, ctx: GameContext, best: Optional[ScoreBreakdown]) -> bool:
        """Decide whether this decision warrants an LLM call.
        
        Play/discard: only for truly desperate situations.
        Shop: handled separately (always uses LLM if enabled).
        """
        if not USE_LLM:
            return False
        if best is None:
            return True
        # Only escalate to LLM if hand is truly terrible AND score is way off
        if best.hand_rank <= 1 and ctx.chips_needed > 0 and best.final_score < ctx.chips_needed * 0.3:
            return True
        return False

    # ============================================================
    # Hand Phase
    # ============================================================

    def decide_hand(self, state: dict) -> Decision:
        """Decide what to do during SELECTING_HAND phase.

        Returns a Decision with action="play" or action="discard".
        """
        ctx = self._build_context(state)

        # Track ante progression
        if ctx.ante > self._last_ante:
            self._last_ante = ctx.ante
            self.archetype.try_commit(ctx.ante)

        if not ctx.hand_cards:
            return Decision("play", {"cards": []}, "No cards in hand", "rule")

        # Find best hands
        best_hands = find_best_hands(ctx.hand_cards, ctx.jokers, ctx.hand_levels, top_n=3)
        if not best_hands:
            indices = list(range(min(5, len(ctx.hand_cards))))
            return Decision("play", {"cards": indices}, "Fallback: play first cards", "rule")

        best = best_hands[0]

        # Rule-based discard check
        do_discard, disc_indices, disc_reason = should_discard(ctx)

        # LLM escalation for complex situations
        if self._should_use_llm(ctx, best) and ctx.discards_left > 0 and ctx.hands_left > 1:
            llm_result = advise_discard(ctx, best)
            if llm_result:
                action = llm_result.get("action", "play")
                reasoning = llm_result.get("reasoning", "LLM decision")

                if action == "discard":
                    cards = llm_result.get("params", {}).get("cards", [])
                    if cards and all(0 <= i < len(ctx.hand_cards) for i in cards):
                        self.archetype.signal_hand(best.hand_type, 0.5)
                        return Decision("discard", {"cards": cards}, reasoning, "llm")

                elif action == "play":
                    cards = llm_result.get("params", {}).get("cards", best.all_cards)
                    if cards and all(0 <= i < len(ctx.hand_cards) for i in cards):
                        self.archetype.signal_hand(best.hand_type)
                        return Decision("play", {"cards": cards}, reasoning, "llm",
                                        score_estimate=best.final_score,
                                        hand_type=best.hand_type)

        # Rule-based decision
        if do_discard:
            return Decision("discard", {"cards": disc_indices}, disc_reason, "rule")

        # Play best hand
        play_indices, play_reason = choose_play(ctx)
        self.archetype.signal_hand(best.hand_type)
        return Decision("play", {"cards": play_indices}, play_reason, "rule",
                        score_estimate=best.final_score,
                        hand_type=best.hand_type)

    # ============================================================
    # Shop Phase
    # ============================================================

    def decide_shop(self, state: dict) -> Decision:
        """Decide what to buy in the shop.

        Uses tier-aware scoring, economy management, and archetype synergy.
        LLM is consulted for nuanced decisions.

        Returns Decision with action="buy", "reroll", or "skip".
        """
        ctx = self._build_context(state)

        if not ctx.shop_items:
            # Check if we should reroll
            do_reroll, reroll_reason = should_reroll(ctx)
            if do_reroll:
                return Decision("reroll", {}, reroll_reason, "rule")
            return Decision("skip", {}, "No items in shop", "rule")

        # Rule-based scoring (now with tier awareness + economy)
        item_scores = shop_decisions(ctx)

        # Lower threshold for high-tier items (S+/S get bought at 4.0+)
        buyable = []
        for idx, score, reason in item_scores:
            if idx >= len(ctx.shop_items):
                continue
            item = ctx.shop_items[idx]
            cost = item.get("cost", 0)
            if cost > ctx.dollars:
                continue
            name = item.get("name", "")
            tier = JOKER_TIERS.get(name)
            # S+ tier: buy at any positive score
            if tier == JokerTier.S_PLUS and score >= 3.0:
                buyable.append((idx, score, reason))
            # S tier: lower threshold
            elif tier == JokerTier.S and score >= 4.0:
                buyable.append((idx, score, reason))
            # Normal threshold
            elif score >= 5.0:
                buyable.append((idx, score, reason))

        # LLM for shop decisions (always complex)
        if USE_LLM and ctx.shop_items:
            llm_result = advise_shop(ctx, item_scores)
            if llm_result:
                action = llm_result.get("action", "skip")
                reasoning = llm_result.get("reasoning", "LLM shop decision")

                if action == "buy":
                    idx = llm_result.get("params", {}).get("index", -1)
                    if 0 <= idx < len(ctx.shop_items):
                        item = ctx.shop_items[idx]
                        cost = item.get("cost", 0)
                        if cost <= ctx.dollars:
                            name = item.get("name", "?")
                            self.archetype.signal_joker(name)
                            return Decision("buy", {"index": idx}, reasoning, "llm")

                return Decision("skip", {}, reasoning, "llm")

        # Rule-based fallback
        if buyable:
            best_idx, best_score, best_reason = buyable[0]
            item = ctx.shop_items[best_idx]
            name = item.get("name", "?")
            self.archetype.signal_joker(name)
            return Decision("buy", {"index": best_idx},
                            f"Buy {name} (score {best_score:.1f}: {best_reason})", "rule")

        # Nothing to buy — consider reroll
        do_reroll, reroll_reason = should_reroll(ctx)
        if do_reroll:
            return Decision("reroll", {}, reroll_reason, "rule")

        return Decision("skip", {}, "Nothing worth buying", "rule")

    # ============================================================
    # Blind Select Phase
    # ============================================================

    def decide_blind(self, state: dict) -> Decision:
        """Decide on blind selection (mainly boss blind strategy).

        Uses knowledge-base counter-strategies for boss blinds,
        with LLM fallback for complex situations.

        Returns Decision with action="select_blind".
        """
        ctx = self._build_context(state)
        blind_info = ctx.blind_info
        boss_name = blind_info.get("boss_name", "")

        if boss_name:
            # Get rule-based counter-strategy from knowledge base
            counter = get_boss_counter(boss_name, ctx)
            reasoning = (
                f"Boss: {boss_name} — {counter['effect']}. "
                f"Counter: {counter['counter']} "
                f"(danger {counter['danger_level']}/3"
                f"{', have counter joker' if counter.get('have_counter') else ''})"
            )

            # Only escalate to LLM for high-danger situations
            if USE_LLM and counter["danger_level"] >= 2:
                llm_result = advise_boss(ctx, boss_name)
                if llm_result:
                    llm_reasoning = llm_result.get("reasoning", "")
                    return Decision("select_blind", {"boss": boss_name},
                                    f"{reasoning} | LLM: {llm_reasoning}", "llm")

            return Decision("select_blind", {"boss": boss_name}, reasoning, "rule")

        return Decision("select_blind", {"boss": boss_name},
                        f"Entering blind (boss: {boss_name or 'none'})", "rule")

    # ============================================================
    # Utility
    # ============================================================

    def record_purchase(self, item_name: str):
        """Record a successful purchase for archetype tracking."""
        self.archetype.signal_joker(item_name)

    def record_planet(self, hand_type: str):
        """Record planet card usage."""
        self.archetype.signal_planet(hand_type)
        level = self.hand_levels.levels.get(hand_type, 1)
        self.hand_levels.levels[hand_type] = level + 1

    def status_summary(self) -> str:
        """Get a human-readable status summary."""
        return (
            f"Build: {self.archetype.archetype_summary()} | "
            f"LLM: {get_llm_stats()}"
        )
