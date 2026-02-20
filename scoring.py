"""Balatro scoring engine — accurate sequential chip/mult calculation.

Implements the real Balatro scoring pipeline (matching EFHIII calculator):
  1. Start with hand-type base chips & base mult (from hand level)
  2. For each scoring card (left to right):
     - Add card's chip value (rank chips + enhancement chips)
     - Apply card enhancement mult effects (+mult or xMult)
     - Apply card edition effects (foil +50 chips, holo +10 mult, poly x1.5)
     - For each joker (left to right): apply per-card-scored triggers
     - If Red Seal: retrigger the card
  3. For each held-in-hand card: apply Steel Card xMult, joker held-card effects
  4. For each joker (left to right):
     - Apply joker's independent scoring effect
     - Apply joker edition (foil/holo/poly)
  5. final_score = chips × mult

Key difference from v1: mult is accumulated SEQUENTIALLY, not separated into
add_mult/x_mult pools. This means joker ORDER matters — matching real Balatro.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from itertools import combinations
from typing import Optional


# ============================================================
# Constants
# ============================================================

RANK_VALUES = {
    "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8,
    "9": 9, "10": 10, "Jack": 10, "Queen": 10, "King": 10, "Ace": 11,
}

RANK_NUM = {
    "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8,
    "9": 9, "10": 10, "Jack": 11, "Queen": 12, "King": 13, "Ace": 14,
}

FACE_RANKS = {"Jack", "Queen", "King"}

# Balatro base scoring for each hand type: (base_chips, base_mult, rank)
HAND_BASE = {
    "Flush Five":       (160, 16, 12),
    "Flush House":      (140, 14, 11),
    "Five of a Kind":   (120, 12, 10),
    "Straight Flush":   (100,  8,  9),
    "Four of a Kind":   ( 60,  7,  8),
    "Full House":       ( 40,  4,  7),
    "Flush":            ( 35,  4,  6),
    "Straight":         ( 30,  4,  5),
    "Three of a Kind":  ( 30,  3,  4),
    "Two Pair":         ( 20,  2,  3),
    "Pair":             ( 10,  2,  2),
    "High Card":        (  5,  1,  1),
}

# Planet card level-up bonuses: (chips_per_level, mult_per_level)
PLANET_BONUS = {
    "Flush Five":       (50, 3),
    "Flush House":      (40, 4),
    "Five of a Kind":   (35, 3),
    "Straight Flush":   (40, 4),
    "Four of a Kind":   (30, 3),
    "Full House":       (25, 2),
    "Flush":            (15, 2),
    "Straight":         (30, 3),
    "Three of a Kind":  (20, 2),
    "Two Pair":         (20, 2),
    "Pair":             (15, 1),
    "High Card":        (10, 1),
}

# Hand types that "contain" sub-types (for joker triggers like Jolly Joker)
HAND_CONTAINS = {
    "Flush Five":      {"Flush Five", "Five of a Kind", "Four of a Kind", "Three of a Kind", "Pair", "Flush"},
    "Flush House":     {"Flush House", "Full House", "Three of a Kind", "Pair", "Flush"},
    "Five of a Kind":  {"Five of a Kind", "Four of a Kind", "Three of a Kind", "Pair"},
    "Straight Flush":  {"Straight Flush", "Straight", "Flush"},
    "Four of a Kind":  {"Four of a Kind", "Three of a Kind", "Pair"},
    "Full House":      {"Full House", "Three of a Kind", "Two Pair", "Pair"},
    "Flush":           {"Flush"},
    "Straight":        {"Straight"},
    "Three of a Kind": {"Three of a Kind", "Pair"},
    "Two Pair":        {"Two Pair", "Pair"},
    "Pair":            {"Pair"},
    "High Card":       {"High Card"},
}


# ============================================================
# Data Types
# ============================================================

@dataclass
class Card:
    """A playing card with all Balatro modifiers."""
    rank: str       # "2"-"10", "Jack", "Queen", "King", "Ace"
    suit: str       # "Hearts", "Diamonds", "Clubs", "Spades"
    enhancement: str = ""
    edition: str = ""
    seal: str = ""
    index: int = 0  # position in hand

    @property
    def chip_value(self) -> int:
        return RANK_VALUES.get(self.rank, 0)

    @property
    def rank_num(self) -> int:
        return RANK_NUM.get(self.rank, 0)

    @property
    def is_face(self) -> bool:
        return self.rank in FACE_RANKS

    @classmethod
    def from_state(cls, data: dict, index: int = 0) -> "Card":
        enh = data.get("enhancement", "")
        if enh in ("Default Base", "Base", ""):
            enh = ""
        return cls(
            rank=data.get("value", data.get("rank", "?")),
            suit=data.get("suit", "?"),
            enhancement=enh,
            edition=data.get("edition", ""),
            seal=data.get("seal", ""),
            index=index,
        )


@dataclass
class Joker:
    """A joker card."""
    name: str
    id: str = ""
    edition: str = ""
    rarity: str = ""
    sell_value: int = 0  # for Swashbuckler etc.

    @classmethod
    def from_state(cls, data: dict) -> "Joker":
        return cls(
            name=data.get("name", "?"),
            id=data.get("id", ""),
            edition=data.get("edition", ""),
            rarity=data.get("rarity", ""),
            sell_value=data.get("sell_value", 0),
        )


@dataclass
class HandLevel:
    """Tracks planet card upgrades for each hand type."""
    levels: dict[str, int] = field(default_factory=lambda: {k: 1 for k in HAND_BASE})
    _game_base: dict[str, tuple[int, int]] = field(default_factory=dict)

    def get_base(self, hand_type: str) -> tuple[int, int]:
        """Return (chips, mult) for a hand type at its current level."""
        if hand_type in self._game_base:
            return self._game_base[hand_type]
        base_chips, base_mult, _ = HAND_BASE.get(hand_type, (5, 1, 1))
        level = self.levels.get(hand_type, 1)
        bonus_chips, bonus_mult = PLANET_BONUS.get(hand_type, (10, 1))
        extra_levels = level - 1
        return (base_chips + bonus_chips * extra_levels,
                base_mult + bonus_mult * extra_levels)

    @classmethod
    def from_game_state(cls, hand_levels_data: dict) -> "HandLevel":
        hl = cls()
        for name, data in hand_levels_data.items():
            if isinstance(data, dict):
                hl.levels[name] = data.get("level", 1)
                chips = data.get("chips", 0)
                mult = data.get("mult", 0)
                if chips > 0 or mult > 0:
                    hl._game_base[name] = (chips, mult)
        return hl


@dataclass
class ScoreBreakdown:
    """Detailed scoring breakdown for a hand."""
    hand_type: str
    hand_rank: int
    base_chips: int
    base_mult: int
    card_chips: int
    add_chips: int      # total added chips (from jokers, editions, enhancements)
    add_mult: int       # total added mult (for backward compat reporting)
    x_mult: float       # product of all xMult sources (for backward compat reporting)
    final_score: float
    scoring_cards: list[int]
    all_cards: list[int]

    @property
    def total_chips(self) -> int:
        return self.base_chips + self.card_chips + self.add_chips

    @property
    def total_mult(self) -> float:
        return (self.base_mult + self.add_mult) * self.x_mult


# ============================================================
# Hand Classification
# ============================================================

def _check_straight(ranks: list[int]) -> bool:
    """Check if ranks form a straight (including A-2-3-4-5 and 10-J-Q-K-A)."""
    s = sorted(set(ranks))
    if len(s) < 5:
        return False
    if s[-1] - s[0] == 4 and len(s) == 5:
        return True
    # Ace-low straight
    if set(s) == {14, 2, 3, 4, 5}:
        return True
    return False


def classify_hand(cards: list[Card]) -> tuple[str, list[int]]:
    """Classify a set of cards into a poker hand type.

    Returns (hand_type, scoring_card_indices).
    """
    if not cards:
        return ("High Card", [])

    n = len(cards)
    ranks = [c.rank_num for c in cards]
    suits = [c.suit for c in cards]
    rc = Counter(ranks).most_common()

    is_flush = len(set(suits)) == 1 and n >= 5
    is_straight = _check_straight(ranks) if n >= 5 else False

    # Five of a Kind
    if rc[0][1] >= 5:
        idxs = [i for i, c in enumerate(cards) if c.rank_num == rc[0][0]][:5]
        if is_flush:
            return ("Flush Five", idxs)
        return ("Five of a Kind", idxs)

    if is_flush and is_straight:
        return ("Straight Flush", list(range(n)))

    if rc[0][1] >= 4:
        quad_rank = rc[0][0]
        quad_idxs = [i for i, c in enumerate(cards) if c.rank_num == quad_rank]
        kicker = [i for i in range(n) if i not in quad_idxs]
        return ("Four of a Kind", quad_idxs + kicker[:1])

    if len(rc) >= 2 and rc[0][1] == 3 and rc[1][1] >= 2:
        trip_rank = rc[0][0]
        pair_rank = rc[1][0]
        trip_idxs = [i for i, c in enumerate(cards) if c.rank_num == trip_rank][:3]
        pair_idxs = [i for i, c in enumerate(cards) if c.rank_num == pair_rank][:2]
        scoring = trip_idxs + pair_idxs
        if is_flush:
            return ("Flush House", scoring)
        return ("Full House", scoring)

    if is_flush:
        return ("Flush", list(range(n)))

    if is_straight:
        return ("Straight", list(range(n)))

    if rc[0][1] == 3:
        trip_rank = rc[0][0]
        idxs = [i for i, c in enumerate(cards) if c.rank_num == trip_rank][:3]
        return ("Three of a Kind", idxs)

    if len(rc) >= 2 and rc[0][1] == 2 and rc[1][1] == 2:
        p1, p2 = rc[0][0], rc[1][0]
        idxs = [i for i, c in enumerate(cards) if c.rank_num in (p1, p2)]
        return ("Two Pair", idxs[:4])

    if rc[0][1] == 2:
        pair_rank = rc[0][0]
        idxs = [i for i, c in enumerate(cards) if c.rank_num == pair_rank][:2]
        return ("Pair", idxs)

    best_idx = max(range(n), key=lambda i: cards[i].rank_num)
    return ("High Card", [best_idx])


# ============================================================
# Sequential Score Calculation (matches real Balatro pipeline)
# ============================================================

class _ScoringContext:
    """Mutable scoring state passed through the pipeline."""
    __slots__ = ('chips', 'mult', 'hand_type', 'hand_contains',
                 'played_cards', 'scoring_idxs', 'held_cards', 'jokers',
                 '_report_add_chips', '_report_add_mult', '_report_x_mult')

    def __init__(self, base_chips: int, base_mult: int, hand_type: str,
                 played_cards: list[Card], scoring_idxs: list[int],
                 held_cards: list[Card] | None, jokers: list[Joker]):
        self.chips = base_chips
        self.mult = float(base_mult)
        self.hand_type = hand_type
        self.hand_contains = HAND_CONTAINS.get(hand_type, {hand_type})
        self.played_cards = played_cards
        self.scoring_idxs = scoring_idxs
        self.held_cards = held_cards or []
        self.jokers = jokers
        # For backward-compat reporting
        self._report_add_chips = 0
        self._report_add_mult = 0
        self._report_x_mult = 1.0

    def add_chips(self, n: int | float):
        self.chips += n
        self._report_add_chips += n

    def add_mult(self, n: int | float):
        self.mult += n
        self._report_add_mult += n

    def x_mult(self, n: float):
        self.mult *= n
        self._report_x_mult *= n


def _trigger_card_scored(ctx: _ScoringContext, card: Card):
    """Process a single scoring card trigger (chips + enhancement + edition + per-card jokers)."""
    # --- Card chip value ---
    if card.enhancement == "Stone Card":
        ctx.add_chips(50)
    else:
        ctx.add_chips(card.chip_value)
        # Bonus Card enhancement
        if card.enhancement == "Bonus Card":
            ctx.add_chips(30)

    # --- Card enhancement mult/xMult ---
    if card.enhancement == "Mult Card":
        ctx.add_mult(4)
    elif card.enhancement == "Glass Card":
        ctx.x_mult(2.0)
    elif card.enhancement == "Lucky Card":
        # Best-case: 1 in 5 chance for +20 mult, 1 in 15 for +$. Use expected value.
        ctx.add_mult(4)  # E[mult] ≈ 20 * (1/5) = 4

    # --- Card edition ---
    if card.edition == "Foil":
        ctx.add_chips(50)
    elif card.edition == "Holographic":
        ctx.add_mult(10)
    elif card.edition == "Polychrome":
        ctx.x_mult(1.5)

    # --- Per-card joker triggers (left to right) ---
    for j in ctx.jokers:
        _joker_on_card_scored(ctx, j, card)


def _joker_on_card_scored(ctx: _ScoringContext, joker: Joker, card: Card):
    """Joker effects that trigger per scoring card (suit/rank bonuses)."""
    name = joker.name

    # Suit-based mult jokers
    if name == "Greedy Joker" and card.suit == "Diamonds":
        ctx.add_mult(3)
    elif name == "Lusty Joker" and card.suit == "Hearts":
        ctx.add_mult(3)
    elif name == "Wrathful Joker" and card.suit == "Spades":
        ctx.add_mult(3)
    elif name == "Gluttonous Joker" and card.suit == "Clubs":
        ctx.add_mult(3)

    # Suit-based chip jokers
    elif name == "Arrowhead" and card.suit == "Spades":
        ctx.add_chips(50)

    # Suit-based mult (uncommon)
    elif name == "Onyx Agate" and card.suit == "Clubs":
        ctx.add_mult(7)

    # Bloodstone: Hearts → 1 in 2 chance x1.5 mult (use expected value)
    elif name == "Bloodstone" and card.suit == "Hearts":
        ctx.x_mult(1.25)  # E[x] = 0.5*1.5 + 0.5*1.0 = 1.25

    # Rough Gem: Diamonds → +$1 (economy, no scoring effect)

    # Face card jokers
    elif name == "Scary Face" and card.is_face:
        ctx.add_chips(30)
    elif name == "Smiley Face" and card.is_face:
        ctx.add_mult(5)

    # Rank-based jokers
    elif name == "Fibonacci" and card.rank in ("Ace", "2", "3", "5", "8"):
        ctx.add_mult(8)
    elif name == "Even Steven" and card.rank in ("2", "4", "6", "8", "10"):
        ctx.add_mult(4)
    elif name == "Odd Todd" and card.rank in ("3", "5", "7", "9", "Ace"):
        ctx.add_chips(31)
    elif name == "Scholar" and card.rank == "Ace":
        ctx.add_chips(20)
        ctx.add_mult(4)
    elif name == "Walkie Talkie" and card.rank in ("10", "4"):
        ctx.add_chips(10)
        ctx.add_mult(4)

    # Triboulet: King or Queen → x2 mult
    elif name == "Triboulet" and card.rank in ("King", "Queen"):
        ctx.x_mult(2.0)

    # Photograph: first face card → x2 mult (simplified: triggers on every face)
    elif name == "Photograph" and card.is_face:
        ctx.x_mult(2.0)


def _trigger_held_card(ctx: _ScoringContext, card: Card):
    """Process a held-in-hand card (Steel Card, joker held-card effects)."""
    if card.enhancement == "Steel Card":
        ctx.x_mult(1.5)

    # Card edition on held cards (only Steel cards trigger)
    if card.enhancement == "Steel Card":
        if card.edition == "Polychrome":
            ctx.x_mult(1.5)
        elif card.edition == "Holographic":
            ctx.add_mult(10)
        elif card.edition == "Foil":
            ctx.add_chips(50)

    # Red Seal on held Steel card → retrigger
    if card.enhancement == "Steel Card" and card.seal == "Red Seal":
        ctx.x_mult(1.5)


def _trigger_joker_independent(ctx: _ScoringContext, joker: Joker):
    """Joker's independent scoring effect (not per-card). Applied left to right."""
    name = joker.name
    hand = ctx.hand_type
    contains = ctx.hand_contains
    played = ctx.played_cards
    scoring = ctx.scoring_idxs
    held = ctx.held_cards
    all_jokers = ctx.jokers

    # --- Flat mult jokers ---
    if name == "Joker":
        ctx.add_mult(4)

    elif name == "Jolly Joker":
        if "Pair" in contains:
            ctx.add_mult(8)
    elif name == "Zany Joker":
        if "Three of a Kind" in contains:
            ctx.add_mult(12)
    elif name == "Mad Joker":
        if "Two Pair" in contains:
            ctx.add_mult(10)
    elif name == "Crazy Joker":
        if "Straight" in contains:
            ctx.add_mult(12)
    elif name == "Droll Joker":
        if "Flush" in contains:
            ctx.add_mult(10)

    elif name == "Half Joker":
        if len(played) <= 3:
            ctx.add_mult(20)

    elif name == "Misprint":
        ctx.add_mult(12)  # avg of 0-23

    elif name == "Mystic Summit":
        ctx.add_mult(8)  # +15 if 0 discards; estimate 50%

    elif name == "Green Joker":
        ctx.add_mult(3)  # +1 per hand played, estimate avg +3

    elif name == "Red Card":
        ctx.add_mult(3)  # +3 per booster skipped, estimate +3

    elif name == "Supernova":
        ctx.add_mult(3)  # +mult = times hand type played, estimate +3

    elif name == "Ride the Bus":
        ctx.add_mult(3)  # +1 per consecutive non-face hand, estimate +3

    elif name == "Swashbuckler":
        total_sell = sum(j.sell_value for j in all_jokers if j.name != "Swashbuckler")
        ctx.add_mult(max(total_sell, 8))  # fallback estimate 8

    elif name == "Abstract Joker":
        ctx.add_mult(3 * len(all_jokers))

    # --- Flat chip jokers ---
    elif name == "Blue Joker":
        ctx.add_chips(60)  # +2 per deck card remaining, estimate ~30

    elif name == "Banner":
        ctx.add_chips(30)  # +30 per discard remaining, estimate 1

    elif name == "Sly Joker":
        if "Pair" in contains:
            ctx.add_chips(50)
    elif name == "Wily Joker":
        if "Three of a Kind" in contains:
            ctx.add_chips(100)
    elif name == "Clever Joker":
        if "Two Pair" in contains:
            ctx.add_chips(80)
    elif name == "Devious Joker":
        if "Straight" in contains:
            ctx.add_chips(100)
    elif name == "Crafty Joker":
        if "Flush" in contains:
            ctx.add_chips(80)

    elif name == "Stuntman":
        ctx.add_chips(250)

    elif name == "Raised Fist":
        if held:
            lowest = min(held, key=lambda c: c.rank_num)
            ctx.add_mult(lowest.rank_num)

    # --- xMult jokers ---
    elif name == "The Duo":
        if "Pair" in contains:
            ctx.x_mult(2.0)
    elif name == "The Trio":
        if "Three of a Kind" in contains:
            ctx.x_mult(3.0)
    elif name == "The Family":
        if "Four of a Kind" in contains:
            ctx.x_mult(4.0)
    elif name == "The Order":
        if "Straight" in contains:
            ctx.x_mult(3.0)
    elif name == "The Tribe":
        if "Flush" in contains:
            ctx.x_mult(2.0)

    elif name == "Stencil Joker":
        empty = max(0, 5 - len(all_jokers))
        if empty > 0:
            ctx.x_mult(1.0 + empty)

    elif name == "Loyalty Card":
        ctx.x_mult(1.2)  # x4 every 6 hands, estimate avg

    elif name == "Acrobat":
        pass  # x3 on final hand — needs round context

    elif name == "Blackboard":
        if held and all(c.suit in ("Spades", "Clubs") for c in held):
            ctx.x_mult(3.0)

    elif name == "Steel Joker":
        if held:
            steel_count = sum(1 for c in held if c.enhancement == "Steel Card")
            if steel_count:
                ctx.x_mult(1.0 + 0.2 * steel_count)

    elif name == "Hiker":
        ctx.add_chips(15)  # +5 permanent per card, estimate avg

    # --- Joker edition (applied after joker's own effect) ---
    # This is handled in calculate_score after calling this function


def calculate_score(
    played_cards: list[Card],
    jokers: list[Joker],
    hand_levels: HandLevel | None = None,
    held_cards: list[Card] | None = None,
) -> ScoreBreakdown:
    """Calculate the score for a played hand with full Balatro mechanics.

    Uses sequential mult accumulation matching real Balatro:
    - Chips accumulate additively
    - Mult accumulates sequentially (add_mult and x_mult interleave by trigger order)
    - Joker order matters

    Args:
        played_cards: Cards being played
        jokers: Active jokers
        hand_levels: Planet card upgrade levels
        held_cards: Cards remaining in hand (for held-card joker effects)

    Returns:
        ScoreBreakdown with full detail
    """
    if hand_levels is None:
        hand_levels = HandLevel()

    hand_type, scoring_idxs = classify_hand(played_cards)
    base_chips, base_mult = hand_levels.get_base(hand_type)
    hand_rank = HAND_BASE.get(hand_type, (5, 1, 1))[2]

    ctx = _ScoringContext(
        base_chips=base_chips,
        base_mult=base_mult,
        hand_type=hand_type,
        played_cards=played_cards,
        scoring_idxs=scoring_idxs,
        held_cards=held_cards,
        jokers=jokers,
    )

    # Phase 1: Score each scoring card (left to right)
    for idx in scoring_idxs:
        card = played_cards[idx]
        _trigger_card_scored(ctx, card)

        # Red Seal retrigger: re-trigger the entire card scoring
        if card.seal == "Red Seal":
            _trigger_card_scored(ctx, card)

    # Phase 2: Held-in-hand card effects
    for card in ctx.held_cards:
        _trigger_held_card(ctx, card)

    # Phase 3: Independent joker effects (left to right, ORDER MATTERS)
    for j in jokers:
        _trigger_joker_independent(ctx, j)

        # Joker edition effects (applied after joker's own effect)
        if j.edition == "Foil":
            ctx.add_chips(50)
        elif j.edition == "Holographic":
            ctx.add_mult(10)
        elif j.edition == "Polychrome":
            ctx.x_mult(1.5)

    # Final score
    final_score = ctx.chips * ctx.mult

    # Card chips = total chips added by cards (not base)
    card_chips = sum(
        (50 if played_cards[i].enhancement == "Stone Card"
         else played_cards[i].chip_value + (30 if played_cards[i].enhancement == "Bonus Card" else 0))
        for i in scoring_idxs
    )

    return ScoreBreakdown(
        hand_type=hand_type,
        hand_rank=hand_rank,
        base_chips=base_chips,
        base_mult=base_mult,
        card_chips=card_chips,
        add_chips=int(ctx._report_add_chips),
        add_mult=int(ctx._report_add_mult),
        x_mult=ctx._report_x_mult,
        final_score=final_score,
        scoring_cards=scoring_idxs,
        all_cards=list(range(len(played_cards))),
    )


# ============================================================
# Hand Finder — find the best hand from a set of cards
# ============================================================

def find_best_hands(
    hand_cards: list[Card],
    jokers: list[Joker],
    hand_levels: HandLevel | None = None,
    held_cards_fn=None,
    max_size: int = 5,
    top_n: int = 3,
) -> list[ScoreBreakdown]:
    """Find the top N scoring hands from available cards.

    Args:
        hand_cards: Cards in hand
        jokers: Active jokers
        hand_levels: Planet upgrade levels
        held_cards_fn: Optional callable(played_indices) -> held_cards
        max_size: Max cards per hand (usually 5)
        top_n: Number of top hands to return

    Returns:
        List of ScoreBreakdown sorted by final_score descending
    """
    if hand_levels is None:
        hand_levels = HandLevel()

    results: list[ScoreBreakdown] = []

    for n in range(min(max_size, len(hand_cards)), 0, -1):
        for combo in combinations(range(len(hand_cards)), n):
            played = [hand_cards[i] for i in combo]
            held = None
            if held_cards_fn:
                held = held_cards_fn(set(combo))
            else:
                held = [hand_cards[i] for i in range(len(hand_cards)) if i not in combo]

            breakdown = calculate_score(played, jokers, hand_levels, held)
            # Map scoring_cards back to original hand indices
            breakdown.all_cards = list(combo)
            breakdown.scoring_cards = [combo[i] for i in breakdown.scoring_cards]
            results.append(breakdown)

    results.sort(key=lambda b: b.final_score, reverse=True)
    return results[:top_n]
