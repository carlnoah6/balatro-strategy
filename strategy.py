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
    FACE_CARDS = "face_cards"  # Baron + Steel Kings
    HIGH_CARD = "high_card"    # High Card spam with scaling jokers
    LUCKY = "lucky"            # Lucky card / probability build
    SCALING = "scaling"        # Hiker / Constellation snowball


# ============================================================
# Joker Tier List (from knowledge base)
# ============================================================

class JokerTier(Enum):
    S_PLUS = "S+"
    S = "S"
    A = "A"
    B = "B"
    C = "C"
    UNKNOWN = "?"

JOKER_TIERS: dict[str, JokerTier] = {
    # S+ Tier — run-defining, always buy
    "Blueprint": JokerTier.S_PLUS,
    "Brainstorm": JokerTier.S_PLUS,
    "Triboulet": JokerTier.S_PLUS,
    # S Tier — build carriers
    "Vampire": JokerTier.S,
    "Cavendish": JokerTier.S,
    "The Duo": JokerTier.S,
    "The Trio": JokerTier.S,
    "The Family": JokerTier.S,
    "Spare Trousers": JokerTier.S,
    "Canio": JokerTier.S,
    "Campfire": JokerTier.S,
    "DNA": JokerTier.S,
    # A Tier — strong picks
    "Hiker": JokerTier.A,
    "Fortune Teller": JokerTier.A,
    "Rocket": JokerTier.A,
    "Seltzer": JokerTier.A,
    "Trading Card": JokerTier.A,
    "Bloodstone": JokerTier.A,
    "Perkeo": JokerTier.A,
    "Hologram": JokerTier.A,
    "Driver's License": JokerTier.A,
    "Steel Joker": JokerTier.A,
    "Card Sharp": JokerTier.A,
    "Shortcut": JokerTier.A,
    "Baron": JokerTier.A,
    "Sock and Buskin": JokerTier.A,
    "Smeared Joker": JokerTier.A,
    "Throwback": JokerTier.A,
    "Oops! All 6s": JokerTier.A,
    # B Tier — solid role players
    "Supernova": JokerTier.B,
    "Scholar": JokerTier.B,
    "Walkie Talkie": JokerTier.B,
    "Wee Joker": JokerTier.B,
    "Square Joker": JokerTier.B,
    "Half Joker": JokerTier.B,
    "Constellation": JokerTier.B,
    "Ceremonial Dagger": JokerTier.B,
    "Blackboard": JokerTier.B,
    "Shoot The Moon": JokerTier.B,
    "Abstract Joker": JokerTier.B,
    "Hack": JokerTier.B,
    "Ride The Bus": JokerTier.B,
    "Green Joker": JokerTier.B,
    "Acrobat": JokerTier.B,
    "Mime": JokerTier.B,
    "Castle": JokerTier.B,
    "Runner": JokerTier.B,
    "Lucky Cat": JokerTier.B,
    "Glass Joker": JokerTier.B,
    "Flower Pot": JokerTier.B,
    "Obelisk": JokerTier.B,
    "Joker Stencil": JokerTier.B,
    # C Tier — situational
    "Matador": JokerTier.C,
    "The Idol": JokerTier.C,
    "Juggler": JokerTier.C,
    "Splash": JokerTier.C,
    "Pareidolia": JokerTier.C,
    "Loyalty Card": JokerTier.C,
    "Dusk": JokerTier.C,
    "Bull": JokerTier.C,
    "Banner": JokerTier.C,
    "Astronomer": JokerTier.C,
    "Drunkard": JokerTier.C,
    "Troubadour": JokerTier.C,
    "Hallucination": JokerTier.C,
    "Chaos The Clown": JokerTier.C,
    "Mr. Bones": JokerTier.C,
    "Merry Andy": JokerTier.C,
    "Red Card": JokerTier.C,
    "Showman": JokerTier.C,
    "Stone Joker": JokerTier.C,
    "Marble Joker": JokerTier.C,
    "Luchador": JokerTier.C,
    "Four Fingers": JokerTier.C,
    "Séance": JokerTier.C,
}

# Tier → score bonus for shop evaluation
TIER_SCORE_BONUS: dict[JokerTier, float] = {
    JokerTier.S_PLUS: 4.0,
    JokerTier.S: 3.0,
    JokerTier.A: 2.0,
    JokerTier.B: 0.5,
    JokerTier.C: -0.5,
    JokerTier.UNKNOWN: 0.0,
}


# ============================================================
# Economy Jokers — jokers that generate money
# ============================================================

ECONOMY_JOKERS = {
    "Rocket", "Golden Joker", "Delayed Gratification", "Business Card",
    "To the Moon", "Satellite", "Cloud 9", "Reserved Parking",
    "Mail-In Rebate", "Hallucination", "Chaos The Clown",
}

# Scaling jokers — buy early for maximum compound value
SCALING_JOKERS = {
    "Hiker", "Constellation", "Wee Joker", "Runner", "Square Joker",
    "Green Joker", "Ride The Bus", "Fortune Teller", "Lucky Cat",
    "Spare Trousers", "Hologram",
}


# ============================================================
# Planet → Hand Type mapping
# ============================================================

PLANET_HAND_MAP = {
    "Pluto": "High Card",
    "Mercury": "Pair",
    "Uranus": "Two Pair",
    "Venus": "Three of a Kind",
    "Saturn": "Straight",
    "Jupiter": "Flush",
    "Earth": "Full House",
    "Mars": "Four of a Kind",
    "Neptune": "Straight Flush",
    "Planet X": "Five of a Kind",
    "Ceres": "Flush House",
    "Eris": "Flush Five",
}

# Reverse: hand type → planet name
HAND_PLANET_MAP = {v: k for k, v in PLANET_HAND_MAP.items()}


# ============================================================
# Boss Blind Counter-Strategies
# ============================================================

BOSS_BLIND_COUNTERS: dict[str, dict] = {
    "The Psychic": {
        "effect": "Must play 5 cards",
        "danger_archetypes": [Archetype.PAIRS, Archetype.HIGH_CARD],
        "counter": "Play 5 cards with your core hand inside. Splash Joker helps.",
        "counter_jokers": ["Splash"],
    },
    "The Pillar": {
        "effect": "Cards played previously are debuffed",
        "danger_archetypes": [Archetype.FOUR_KIND],
        "counter": "Vary your played cards. Large/varied deck helps.",
        "counter_jokers": [],
    },
    "The Mark": {
        "effect": "Face cards drawn face down",
        "danger_archetypes": [Archetype.FACE_CARDS],
        "counter": "Avoid relying on face card identification.",
        "counter_jokers": ["Luchador"],
    },
    "The Plant": {
        "effect": "Face cards are debuffed",
        "danger_archetypes": [Archetype.FACE_CARDS],
        "counter": "Hard counter to face builds. Reroll or skip.",
        "counter_jokers": ["Luchador"],
    },
    "The Fish": {
        "effect": "Cards drawn face down",
        "danger_archetypes": [],
        "counter": "Play hands that work regardless of card visibility.",
        "counter_jokers": ["Luchador"],
    },
    "The Verdant": {
        "effect": "Debuffs a specific suit (Clubs)",
        "danger_archetypes": [Archetype.FLUSH],
        "counter": "Keep 2+ suits viable. Smeared Joker merges suits.",
        "counter_jokers": ["Smeared Joker", "Luchador"],
    },
    "The Crimson": {
        "effect": "Debuffs a specific suit (Hearts)",
        "danger_archetypes": [Archetype.FLUSH],
        "counter": "Keep 2+ suits viable. Smeared Joker merges suits.",
        "counter_jokers": ["Smeared Joker", "Luchador"],
    },
    "The Violet": {
        "effect": "Debuffs a specific suit (Spades)",
        "danger_archetypes": [Archetype.FLUSH],
        "counter": "Keep 2+ suits viable. Smeared Joker merges suits.",
        "counter_jokers": ["Smeared Joker", "Luchador"],
    },
    "The Amber": {
        "effect": "Debuffs a specific suit (Diamonds)",
        "danger_archetypes": [Archetype.FLUSH],
        "counter": "Keep 2+ suits viable. Smeared Joker merges suits.",
        "counter_jokers": ["Smeared Joker", "Luchador"],
    },
}

# Key vouchers worth buying
PRIORITY_VOUCHERS = {
    "Director's Cut", "Reroll Surplus",  # Reroll boss blinds
    "Overstock", "Overstock Plus",       # More shop cards
    "Hone", "Glow Up",                  # Better edition odds
    "Money Tree", "Seed Money",          # Raise interest cap
    "Blank", "Antimatter",               # Extra joker slot
}


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
    Archetype.FACE_CARDS: {
        "Baron", "Mime", "Triboulet", "Sock and Buskin", "Pareidolia",
    },
    Archetype.HIGH_CARD: {
        "Supernova", "Green Joker", "Square Joker", "Card Sharp",
    },
    Archetype.LUCKY: {
        "Oops! All 6s", "Lucky Cat", "Bloodstone", "Business Card",
    },
    Archetype.SCALING: {
        "Hiker", "Runner", "Constellation", "Wee Joker", "Square Joker",
    },
}

# Hand types that signal an archetype
ARCHETYPE_HANDS = {
    Archetype.FLUSH: {"Flush", "Straight Flush", "Flush Five", "Flush House"},
    Archetype.PAIRS: {"Pair", "Two Pair", "Full House", "Flush House"},
    Archetype.STRAIGHT: {"Straight", "Straight Flush"},
    Archetype.FOUR_KIND: {"Four of a Kind", "Five of a Kind", "Flush Five"},
    Archetype.HIGH_CARD: {"High Card"},
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

    Improved with:
    - Better archetype-aware card retention (flush, pairs, straight, face_cards, etc.)
    - Enhanced card priority scoring (keep high-value enhancements)
    - Boss blind awareness (conserve discards for boss rounds)
    - Smarter aggression based on hands remaining vs score deficit

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

    # Boss blind round: be more conservative with discards (save for later hands)
    is_boss = ctx.round_num == 2 or bool(ctx.blind_info.get("boss_name"))
    discard_budget = ctx.discards_left
    if is_boss and ctx.hands_left >= 3 and best.hand_rank >= 4:
        # Decent hand on boss round — save discards for later
        return (False, [], f"Boss round — save discards, hand is {best.hand_type}")

    # Calculate discard value: what cards are NOT in the best hand?
    best_indices = set(best.all_cards)
    non_scoring = [i for i in range(len(ctx.hand_cards)) if i not in best_indices]

    # Archetype-aware discard: score each non-scoring card for "keep value"
    arch = ctx.archetype.current
    discard_candidates = []

    # Pre-compute hand statistics
    suit_counts = Counter(c.suit for c in ctx.hand_cards)
    rank_counts = Counter(c.rank for c in ctx.hand_cards)
    dominant_suit = suit_counts.most_common(1)[0][0] if suit_counts else ""
    all_ranks = sorted(set(c.rank_num for c in ctx.hand_cards))

    for i in non_scoring:
        card = ctx.hand_cards[i]
        keep_score = 0.0  # higher = more reason to keep

        # Always keep enhanced/edition/seal cards (they have permanent value)
        if card.enhancement:
            keep_score += 3.0
            if card.enhancement in ("Steel", "Glass", "Gold"):
                keep_score += 2.0  # premium enhancements
        if card.edition:
            keep_score += 2.0
        if card.seal:
            keep_score += 2.0

        # Archetype-specific retention
        if arch == Archetype.FLUSH:
            if card.suit == dominant_suit:
                keep_score += 2.0
        elif arch in (Archetype.PAIRS, Archetype.FOUR_KIND):
            if rank_counts[card.rank] >= 2:
                keep_score += 3.0  # part of a pair/set
        elif arch == Archetype.STRAIGHT:
            if _contributes_to_straight(card.rank_num, all_ranks):
                keep_score += 2.0
        elif arch == Archetype.FACE_CARDS:
            if card.rank in ("Jack", "Queen", "King"):
                keep_score += 3.0
        elif arch == Archetype.HIGH_CARD:
            if card.rank == "Ace":
                keep_score += 1.0

        # High-rank cards have marginal value (more chips when scored)
        if card.rank_num >= 10:
            keep_score += 0.5

        if keep_score < 2.0:
            discard_candidates.append((i, keep_score))

    # Sort by keep_score ascending (discard lowest value first)
    discard_candidates.sort(key=lambda x: x[1])
    discard_indices = [i for i, _ in discard_candidates]

    if not discard_indices:
        # Nothing obvious to discard — check if hand is weak enough to warrant it
        if best.hand_rank <= 3 and ctx.hands_left > 1:
            # Weak hand (High Card / Pair / Two Pair), discard non-scoring cards aggressively
            discard_indices = non_scoring[:min(5, discard_budget)]
        elif best.final_score < chips_needed * 0.7 and ctx.hands_left > 1:
            # Score too low for target — discard non-scoring to try for better
            discard_indices = non_scoring[:min(3, discard_budget)]
        else:
            return (False, [], f"Hand is decent ({best.hand_type}), no clear discards")

    # Limit to available discards
    to_discard = discard_indices[:discard_budget]

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

def _has_economy_joker(ctx: GameContext) -> bool:
    """Check if we already have an economy joker."""
    return any(j.name in ECONOMY_JOKERS for j in ctx.jokers)


def _count_xmult_jokers(ctx: GameContext) -> int:
    """Count jokers that provide xMult."""
    xmult_names = {
        "Cavendish", "The Duo", "The Trio", "The Family", "The Order",
        "The Tribe", "Bloodstone", "Card Sharp", "Oops! All 6s",
        "Driver's License", "Steel Joker", "Glass Joker", "Acrobat",
        "Baron", "Hologram", "Lucky Cat", "Vampire", "Campfire",
        "Blueprint", "Brainstorm", "Triboulet",
    }
    return sum(1 for j in ctx.jokers if j.name in xmult_names)


def evaluate_shop_item(item: dict, ctx: GameContext) -> tuple[float, str]:
    """Score a shop item from 0-10 based on strategic value.

    Incorporates joker tier awareness, economy management, planet
    prioritization, and archetype synergy from the knowledge base.

    Returns (score, reasoning).
    """
    name = item.get("name", "")
    cost = item.get("cost", 0)
    item_type = item.get("type", "")
    edition = item.get("edition", "")

    if cost > ctx.dollars:
        return (0.0, "Can't afford")

    score = 5.0  # baseline
    reasons = []

    # ── Economy guard ──────────────────────────────────────────
    # Protect the $25 interest threshold (max $5/round)
    money_after = ctx.dollars - cost
    interest_before = min(ctx.dollars // 5, 5)
    interest_after = min(money_after // 5, 5)
    interest_loss = interest_before - interest_after

    if interest_loss > 0 and ctx.ante >= 2:
        # Losing interest is costly — $1/round compounds over the run
        penalty = interest_loss * 2.0
        score -= penalty
        reasons.append(f"loses ${interest_loss}/round interest")

    # Hard rule: never drop below $15 in ante 2-4 unless item is S/S+ tier
    tier = JOKER_TIERS.get(name, JokerTier.UNKNOWN)
    if money_after < 15 and 2 <= ctx.ante <= 4 and tier not in (JokerTier.S_PLUS, JokerTier.S):
        score -= 2.0
        reasons.append("would break economy floor ($15)")

    # ── Joker evaluation ───────────────────────────────────────
    if item_type == "Joker":
        if ctx.joker_space <= 0:
            # Negative edition doesn't use a slot
            if edition != "Negative":
                return (0.0, "No joker slots")
            else:
                score += 1.0
                reasons.append("Negative edition — free slot")

        # Tier-based scoring (highest impact improvement)
        tier_bonus = TIER_SCORE_BONUS.get(tier, 0.0)
        if tier_bonus != 0:
            score += tier_bonus
            reasons.append(f"{tier.value}-tier joker")

        # S+ tier: always buy if affordable (override economy concerns)
        if tier == JokerTier.S_PLUS:
            score = max(score, 9.0)
            reasons.append("RUN-DEFINING — always buy")

        # Edition bonus (Polychrome > Holographic > Foil)
        if edition == "Polychrome":
            score += 2.0
            reasons.append("Polychrome edition (×1.5)")
        elif edition == "Holographic":
            score += 1.5
            reasons.append("Holographic edition (+10 mult)")
        elif edition == "Foil":
            score += 0.5
            reasons.append("Foil edition (+50 chips)")
        elif edition == "Negative":
            score += 2.5
            reasons.append("Negative edition — no slot used")

        # Early game: need jokers to survive
        num_jokers = len(ctx.jokers)
        if ctx.ante <= 3 and num_jokers < 3:
            score += 2.0
            reasons.append("early game, need jokers")

        # Economy joker awareness
        if name in ECONOMY_JOKERS:
            if not _has_economy_joker(ctx):
                score += 1.5
                reasons.append("first economy joker")
            elif ctx.ante <= 2:
                score += 0.5
                reasons.append("extra economy early")

        # Scaling jokers: buy early for compound value, penalize late
        if name in SCALING_JOKERS:
            if ctx.ante <= 2:
                score += 2.0
                reasons.append("scaling joker — early = max compound")
            elif ctx.ante <= 4:
                score += 1.0
                reasons.append("scaling joker — still good mid-game")
            else:
                score -= 1.0
                reasons.append("scaling joker — too late to compound")

        # xMult awareness: need at least 1 by ante 4, 2+ by ante 6
        xmult_count = _count_xmult_jokers(ctx)
        xmult_names = {
            "Cavendish", "The Duo", "The Trio", "The Family", "The Order",
            "The Tribe", "Bloodstone", "Card Sharp", "Oops! All 6s",
            "Driver's License", "Steel Joker", "Glass Joker", "Acrobat",
            "Baron", "Hologram", "Lucky Cat", "Vampire", "Campfire",
        }
        if name in xmult_names:
            if xmult_count == 0 and ctx.ante >= 3:
                score += 2.5
                reasons.append("NEED xMult — first one")
            elif xmult_count < 2 and ctx.ante >= 5:
                score += 2.0
                reasons.append("need more xMult for late game")
            else:
                score += 1.0
                reasons.append("xMult source")

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
                    # Off-archetype but high tier is still worth considering
                    if tier in (JokerTier.S_PLUS, JokerTier.S):
                        score += 0.5
                        reasons.append(f"off-archetype but high tier")
                    else:
                        score -= 1.0
                        reasons.append(f"off-archetype ({a.value})")
                break

        # Universal formula check: 1 econ + 1-2 scaling + 1 utility + 2-3 xMult
        # Gently nudge toward filling gaps
        if num_jokers >= 3 and xmult_count == 0 and name not in xmult_names:
            score -= 0.5
            reasons.append("have jokers but no xMult yet")

    # ── Planet card evaluation ─────────────────────────────────
    elif item_type == "Planet":
        if ctx.consumable_space <= 0:
            return (0.0, "No consumable slots")

        # Map planet name to hand type
        planet_hand = PLANET_HAND_MAP.get(name, "")
        arch = ctx.archetype.current

        if planet_hand:
            # Check if this planet matches our archetype's preferred hands
            arch_hands = ARCHETYPE_HANDS.get(arch, set())
            if planet_hand in arch_hands:
                score += 3.0
                reasons.append(f"levels {planet_hand} — core hand for {arch.value}")
            elif arch == Archetype.UNDECIDED:
                # Check what we've been playing
                if ctx.archetype.hand_history:
                    recent = ctx.archetype.hand_history[-5:]
                    if planet_hand in recent:
                        score += 2.0
                        reasons.append(f"levels {planet_hand} — recently played")
                    else:
                        score += 0.5
                        reasons.append(f"levels {planet_hand}")
                else:
                    score += 1.0
                    reasons.append(f"levels {planet_hand}")
            else:
                score -= 0.5
                reasons.append(f"levels {planet_hand} — not our build")
        else:
            score += 1.0
            reasons.append("planet card")

    # ── Tarot evaluation ───────────────────────────────────────
    elif item_type == "Tarot":
        if ctx.consumable_space <= 0:
            return (0.0, "No consumable slots")
        score += 0.5
        reasons.append("tarot card")
        # Suit-changing tarots are great for flush builds
        arch = ctx.archetype.current
        if arch == Archetype.FLUSH:
            suit_tarots = {"Lovers", "Empress", "Emperor", "Hierophant"}
            if name in suit_tarots:
                score += 2.0
                reasons.append("suit conversion for flush build")

    # ── Voucher evaluation ─────────────────────────────────────
    elif item_type == "Voucher":
        if name in PRIORITY_VOUCHERS:
            score += 2.5
            reasons.append(f"priority voucher")
        else:
            score += 0.5
            reasons.append("voucher")

    # ── Game phase adjustments ─────────────────────────────────
    # Early game: aggressive rerolling is correct, but don't overspend
    if ctx.ante <= 2 and cost > 4 and tier not in (JokerTier.S_PLUS, JokerTier.S, JokerTier.A):
        score -= 1.0
        reasons.append("expensive for early game")

    # Mid game: need xMult sources
    if 4 <= ctx.ante <= 6 and item_type == "Joker":
        if _count_xmult_jokers(ctx) == 0:
            score += 0.5
            reasons.append("mid-game — any joker helps find xMult")

    # Late game: power matters more than economy
    if ctx.ante >= 6:
        score += 1.0
        reasons.append("late game — power matters more")
        # Reduce economy penalty in late game
        if interest_loss > 0:
            score += interest_loss * 0.5  # partially offset the penalty
            reasons.append("economy less critical late")

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


# ============================================================
# Boss Blind Strategy
# ============================================================

def get_boss_counter(boss_name: str, ctx: GameContext) -> dict:
    """Get counter-strategy info for a boss blind.

    Returns dict with keys: effect, counter, danger_level (0-3),
    counter_jokers, and strategy_notes.
    """
    info = BOSS_BLIND_COUNTERS.get(boss_name, {})
    if not info:
        return {
            "effect": "Unknown boss",
            "counter": "Play normally",
            "danger_level": 0,
            "counter_jokers": [],
            "strategy_notes": "",
        }

    arch = ctx.archetype.current
    danger_archetypes = info.get("danger_archetypes", [])
    danger_level = 2 if arch in danger_archetypes else 1

    # Check if we have counter jokers
    counter_jokers = info.get("counter_jokers", [])
    have_counter = any(j.name in counter_jokers for j in ctx.jokers)
    if have_counter:
        danger_level = max(0, danger_level - 1)

    # Extra danger if we're low on hands/discards
    if ctx.hands_left <= 2:
        danger_level = min(3, danger_level + 1)

    return {
        "effect": info.get("effect", ""),
        "counter": info.get("counter", ""),
        "danger_level": danger_level,
        "counter_jokers": counter_jokers,
        "have_counter": have_counter,
        "strategy_notes": f"Build: {arch.value}, danger: {danger_level}/3",
    }


# ============================================================
# Economy Strategy Helpers
# ============================================================

def should_reroll(ctx: GameContext) -> tuple[bool, str]:
    """Decide whether to spend $5 to reroll the shop.

    Returns (should_reroll, reasoning).
    """
    # Never reroll if it would break interest threshold in mid-game
    if ctx.ante >= 2 and ctx.dollars - 5 < 25 and ctx.dollars >= 25:
        return (False, "Would break $25 interest threshold")

    # Aggressive rerolling is correct in ante 1-2
    if ctx.ante <= 2 and ctx.dollars >= 10:
        num_jokers = len(ctx.jokers)
        if num_jokers < 2:
            return (True, "Early game — need jokers, can afford reroll")

    # Late game: reroll if we have excess money and need xMult
    if ctx.ante >= 5 and ctx.dollars >= 35 and _count_xmult_jokers(ctx) < 2:
        return (True, "Late game — excess money, need xMult")

    return (False, "Save money")
