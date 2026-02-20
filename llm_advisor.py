"""LLM advisor for complex Balatro decisions.

Provides structured prompts with full game context, archetype awareness,
and knowledge base integration. Falls back to rule-based decisions on failure.
"""

from __future__ import annotations

import json
import os
import time
from typing import Optional

import requests

from .scoring import Card, Joker, HandLevel, ScoreBreakdown, find_best_hands
from .strategy import GameContext, Archetype, ArchetypeTracker

LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "http://localhost:8180/v1")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "sk-luna-2026-openclaw")
LLM_MODEL = os.environ.get("LLM_MODEL", "kimi-k2.5")

# Track LLM call stats
_llm_stats = {"calls": 0, "failures": 0, "total_ms": 0, "input_tokens": 0, "output_tokens": 0}


def get_llm_stats() -> dict:
    return dict(_llm_stats)


# ============================================================
# System Prompt
# ============================================================

SYSTEM_PROMPT = """You are an expert Balatro player AI making real-time decisions.
You MUST write all reasoning in Chinese (中文). JSON keys stay in English.

## Scoring Formula
final_score = (base_chips + card_chips) × (base_mult + add_mult) × product(xMult)

Each hand type has base chips/mult (upgradeable via Planet cards):
- High Card: 5×1 | Pair: 10×2 | Two Pair: 20×2 | Three of a Kind: 30×3
- Straight: 30×4 | Flush: 35×4 | Full House: 40×4 | Four of a Kind: 60×7
- Straight Flush: 100×8 | Five of a Kind: 120×12

## Key Mechanics
- Jokers trigger left-to-right; order matters for conditional effects
- xMult sources multiply together (1.5 × 2.0 = 3.0x)
- Interest: $1 per $5 saved, max $5 at $25+. Protect the $25 threshold.
- Enhancements: Bonus(+30 chips), Mult(+4 mult), Glass(×2 but can break), Steel(×1.5 while held), Stone(+50 chips, no rank/suit), Gold(+$3 end of round)
- Editions: Foil(+50 chips), Holographic(+10 mult), Polychrome(×1.5)
- Red Seal: retrigger card scoring

## Decision Framework
1. Can I clear this blind with what I have? → Play the minimum hand needed
2. Is discarding worth the risk? → Only if expected improvement > current hand value
3. Shop: Does this item synergize with my build? Will buying break my interest threshold?
4. Boss blind: What specific counter-strategy is needed?

## CRITICAL: Response Format
You MUST respond with ONLY a JSON object. No markdown, no extra text.
Keep reasoning under 100 characters. Example:
{"action": "discard", "params": {"cards": [2, 5, 7]}, "reasoning": "弃掉三张废牌，保留对子骨架"}
{"action": "play", "params": {"cards": [0, 1, 3, 5, 6]}, "reasoning": "打出同花，超额1.5倍"}
{"action": "buy", "params": {"index": 0}, "reasoning": "买Joker强化对子路线"}
{"action": "skip", "reasoning": "保留利息，商店没有好东西"}
"""


# ============================================================
# Prompt Builders
# ============================================================

def _format_cards(cards: list[Card], label: str = "Hand") -> str:
    lines = [f"{label}:"]
    for i, c in enumerate(cards):
        extras = []
        if c.enhancement:
            extras.append(f"[{c.enhancement}]")
        if c.edition:
            extras.append(f"({c.edition})")
        if c.seal:
            extras.append(f"<{c.seal}>")
        extra_str = " ".join(extras)
        lines.append(f"  [{i}] {c.rank} of {c.suit} {extra_str}".rstrip())
    return "\n".join(lines)


def _format_jokers(jokers: list[Joker]) -> str:
    if not jokers:
        return "Jokers: (none)"
    lines = ["Jokers (trigger left→right):"]
    for i, j in enumerate(jokers):
        ed = f" ({j.edition})" if j.edition else ""
        lines.append(f"  [{i}] {j.name}{ed}")
    return "\n".join(lines)


def _format_context(ctx: GameContext) -> str:
    """Build the full context block for any decision."""
    parts = [
        f"Ante: {ctx.ante} | Blind target: {ctx.blind_chips:,.0f} | Current score: {ctx.current_chips:,.0f}",
        f"Hands left: {ctx.hands_left} | Discards left: {ctx.discards_left} | Money: ${ctx.dollars}",
        f"Build: {ctx.archetype.archetype_summary()}",
        "",
        _format_jokers(ctx.jokers),
    ]
    if ctx.hand_cards:
        parts.append("")
        parts.append(_format_cards(ctx.hand_cards))
    return "\n".join(parts)


def build_discard_prompt(ctx: GameContext, best_hand: ScoreBreakdown) -> str:
    """Build a focused discard decision prompt."""
    return f"""{_format_context(ctx)}

Best current hand: {best_hand.hand_type} (cards {best_hand.all_cards}) → {best_hand.final_score:,.0f} score
Score still needed: {ctx.chips_needed:,.0f}

Question: Should I discard to improve my hand?
- If yes: which card indices to discard? Consider what I might draw.
- If no: I'll play the best hand.

Respond: {{"action": "discard", "params": {{"cards": [indices]}}, "reasoning": "..."}}
or: {{"action": "play", "params": {{"cards": {best_hand.all_cards}}}, "reasoning": "..."}}"""


def build_shop_prompt(ctx: GameContext, item_scores: list[tuple[int, float, str]]) -> str:
    """Build a shop purchase decision prompt."""
    items_str = ""
    for idx, score, reason in item_scores:
        if idx < len(ctx.shop_items):
            item = ctx.shop_items[idx]
            items_str += f"  [{idx}] {item.get('name','?')} (${item.get('cost',0)}, {item.get('type','?')}) — rule score: {score:.1f} ({reason})\n"

    return f"""{_format_context(ctx)}

Joker slots: {len(ctx.jokers)}/{ctx.joker_slots}
Consumable slots: {len(ctx.consumables)}/{ctx.consumable_slots}
Interest per round: ${ctx.interest_money}

Shop items (with rule-based scores):
{items_str}
Question: Should I buy anything? Consider:
1. Archetype synergy with my {ctx.archetype.current.value} build
2. Economy impact (will I lose interest?)
3. Joker slot availability

Respond: {{"action": "buy", "params": {{"index": N}}, "reasoning": "..."}}
or: {{"action": "skip", "reasoning": "..."}}"""


def build_boss_prompt(ctx: GameContext, boss_name: str) -> str:
    """Build a boss blind strategy prompt."""
    return f"""{_format_context(ctx)}

Boss Blind: {boss_name}

Question: What strategy should I use against this boss?
Consider which cards/jokers counter the boss effect.

Respond: {{"action": "select_blind", "params": {{}}, "reasoning": "strategy explanation"}}"""


# ============================================================
# LLM Call
# ============================================================

def call_llm(prompt: str, timeout: float = 30.0) -> Optional[dict]:
    """Call the LLM and parse JSON response.

    Returns parsed dict or None on failure.
    """
    _llm_stats["calls"] += 1
    start = time.time()

    try:
        r = requests.post(
            f"{LLM_BASE_URL}/chat/completions",
            headers={
                "Authorization": f"Bearer {LLM_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": LLM_MODEL,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.2,
                "max_tokens": 1024,
            },
            timeout=timeout,
        )
        data = r.json()
        content = data["choices"][0]["message"]["content"]

        elapsed_ms = (time.time() - start) * 1000
        _llm_stats["total_ms"] += elapsed_ms

        # Track token usage
        usage = data.get("usage", {})
        _llm_stats["input_tokens"] += usage.get("prompt_tokens", 0)
        _llm_stats["output_tokens"] += usage.get("completion_tokens", 0)

        return _parse_json_response(content)

    except Exception as e:
        _llm_stats["failures"] += 1
        elapsed_ms = (time.time() - start) * 1000
        _llm_stats["total_ms"] += elapsed_ms
        print(f"[llm_advisor] Error ({elapsed_ms:.0f}ms): {e}")
        return None


def _call_llm_raw(prompt: str, max_tokens: int = 512, timeout: float = 30.0) -> Optional[str]:
    """Call LLM and return raw text response (no JSON parsing)."""
    try:
        r = requests.post(
            f"{LLM_BASE_URL}/chat/completions",
            headers={"Authorization": f"Bearer {LLM_API_KEY}", "Content-Type": "application/json"},
            json={"model": LLM_MODEL, "messages": [{"role": "user", "content": prompt}],
                  "temperature": 0.3, "max_tokens": max_tokens},
            timeout=timeout,
        )
        return r.json()["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"[llm_advisor] _call_llm_raw error: {e}")
        return None


def _parse_json_response(content: str) -> Optional[dict]:
    """Extract JSON from LLM response text."""
    # Try direct parse
    try:
        return json.loads(content.strip())
    except json.JSONDecodeError:
        pass

    # Try extracting from markdown code block
    if "```json" in content:
        content = content.split("```json")[1].split("```")[0]
    elif "```" in content:
        content = content.split("```")[1].split("```")[0]

    try:
        return json.loads(content.strip())
    except json.JSONDecodeError:
        pass

    # Try finding JSON object
    start = content.find("{")
    end = content.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            return json.loads(content[start:end])
        except json.JSONDecodeError:
            pass

    # Try repairing truncated JSON (missing closing braces/brackets)
    if start >= 0:
        fragment = content[start:]
        # Count unmatched braces/brackets
        opens = fragment.count("{") - fragment.count("}")
        open_brackets = fragment.count("[") - fragment.count("]")
        repaired = fragment + "]" * max(0, open_brackets) + "}" * max(0, opens)
        try:
            return json.loads(repaired)
        except json.JSONDecodeError:
            pass

    # Last resort: extract action and params with regex
    import re
    if start >= 0:
        fragment = content[start:]
        action_m = re.search(r'"action"\s*:\s*"(\w+)"', fragment)
        if action_m:
            result = {"action": action_m.group(1)}
            # Extract cards array
            cards_m = re.search(r'"cards"\s*:\s*\[([\d,\s]*)', fragment)
            if cards_m:
                try:
                    cards = [int(x.strip()) for x in cards_m.group(1).split(",") if x.strip().isdigit()]
                    result["params"] = {"cards": cards}
                except ValueError:
                    pass
            # Extract index
            index_m = re.search(r'"index"\s*:\s*(\d+)', fragment)
            if index_m:
                result["params"] = {"index": int(index_m.group(1))}
            # Extract reasoning (best effort)
            reason_m = re.search(r'"reasoning"\s*:\s*"([^"]*)', fragment)
            if reason_m:
                result["reasoning"] = reason_m.group(1)
            return result

    print(f"[llm_advisor] Could not parse: {content[:200]}")
    return None


# ============================================================
# High-Level Decision Functions
# ============================================================

def advise_discard(ctx: GameContext, best_hand: ScoreBreakdown) -> Optional[dict]:
    """Ask LLM whether to discard. Returns parsed response or None."""
    prompt = build_discard_prompt(ctx, best_hand)
    return call_llm(prompt)


def advise_shop(ctx: GameContext, item_scores: list[tuple[int, float, str]]) -> Optional[dict]:
    """Ask LLM what to buy in shop. Returns parsed response or None."""
    prompt = build_shop_prompt(ctx, item_scores)
    return call_llm(prompt)


def advise_boss(ctx: GameContext, boss_name: str) -> Optional[dict]:
    """Ask LLM for boss blind strategy. Returns parsed response or None."""
    prompt = build_boss_prompt(ctx, boss_name)
    return call_llm(prompt)
