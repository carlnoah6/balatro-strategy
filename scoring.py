"""Balatro scoring engine — accurate chip×mult×xMult calculation.

Implements the real Balatro scoring formula:
  final_score = (base_chips + card_chips) × (base_mult + add_mult) × product(xMult_sources)

Card chips come from played cards. Jokers, enhancements, seals, and editions
modify chips, mult, or xMult at various stages.
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

# Enhancement effects
ENHANCEMENT_CHIPS = {
    "Bonus Card": 30,
    "Stone Card": 50,
}
ENHANCEMENT_MULT = {
    "Mult Card": 4,
}
ENHANCEMENT_XMULT = {
    "Glass Card": 2.0,
}
ENHANCEMENT_GOLD_PAYOUT = 3  # Gold Card: +$3 at end of round

# Edition effects
EDITION_CHIPS = {"Foil": 50}
EDITION_MULT = {"Holographic": 10}
EDITION_XMULT = {"Polychrome": 1.5}

# Seal effects (simplified)
SEAL_RETRIGGER = {"Red Seal"}  # retrigger card scoring


# ============================================================
# Data Types
# ============================================================

@dataclass
class Card:
    """A playing card with all Balatro modifiers."""
    rank: str  # "2"-"10", "Jack", "Queen", "King", "Ace"
    suit: str  # "Hearts", "Diamonds", "Clubs", "Spades"
    enhancement: str = ""
    edition: str = ""
    seal: str = ""
    index: int = 0  # position in hand

    @property
    def chip_value(self) -> int:
        return RANK_VALUES.get(self.rank, 0)

    @property
    def rank_num(self) -> int:
        return {"2":2,"3":3,"4":4,"5":5,"6":6,"7":7,"8":8,
                "9":9,"10":10,"Jack":11,"Queen":12,"King":13,"Ace":14}.get(self.rank, 0)

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

    @classmethod
    def from_state(cls, data: dict) -> "Joker":
        return cls(
            name=data.get("name", "?"),
            id=data.get("id", ""),
            edition=data.get("edition", ""),
            rarity=data.get("rarity", ""),
        )


@dataclass
class HandLevel:
    """Tracks planet card upgrades for each hand type."""
    levels: dict[str, int] = field(default_factory=lambda: {k: 1 for k in HAND_BASE})
    # Override chips/mult from game state (more accurate than calculating from levels)
    _game_base: dict[str, tuple[int, int]] = field(default_factory=dict)

    def get_base(self, hand_type: str) -> tuple[int, int]:
        """Return (chips, mult) for a hand type at its current level."""
        # Prefer game-reported values (includes all planet upgrades accurately)
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
        """Build from Lua game state hand_levels dict.
        
        Each entry: {level, chips, mult, played}
        Uses the game's actual chips/mult values directly.
        """
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
    add_chips: int  # from jokers, enhancements
    add_mult: int   # from jokers, enhancements
    x_mult: float   # product of all xMult sources
    final_score: float
    scoring_cards: list[int]  # indices of cards that scored
    all_cards: list[int]      # indices of all played cards

    @property
    def total_chips(self) -> int:
        return self.base_chips + self.card_chips + self.add_chips

    @property
    def total_mult(self) -> float:
        return (self.base_mult + self.add_mult) * self.x_mult


# ============================================================
# Hand Classification
# ============================================================

def classify_hand(cards: list[Card]) -> tuple[str, list[int]]:
    """Classify a set of cards into a poker hand type.

    Returns (hand_type, scoring_card_indices).
    scoring_card_indices are the cards that actually contribute to scoring.
    """
    if not cards:
        return ("High Card", [])

    n = len(cards)
    ranks = [c.rank_num for c in cards]
    suits = [c.suit for c in cards]
    rc = Counter(ranks).most_common()

    is_flush = len(set(suits)) == 1 and n >= 5
    is_straight = _check_straight(ranks) if n >= 5 else False

    # Five of a Kind (requires special joker like Smeared Joker)
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

    # High Card — only the highest card scores
    best_idx = max(range(n), key=lambda i: cards[i].rank_num)
    return ("High Card", [best_idx])


def _check_straight(ranks: list[int]) -> bool:
    """Check if ranks form a straight (including A-2-3-4-5)."""
    s = sorted(set(ranks))
    if len(s) < 5:
        return False
    # Normal straight
    if s[-1] - s[0] == 4 and len(s) == 5:
        return True
    # Ace-low straight (A-2-3-4-5)
    if set(s) == {14, 2, 3, 4, 5}:
        return True
    return False


# ============================================================
# Score Calculation
# ============================================================

def calculate_score(
    played_cards: list[Card],
    jokers: list[Joker],
    hand_levels: HandLevel | None = None,
    held_cards: list[Card] | None = None,
) -> ScoreBreakdown:
    """Calculate the score for a played hand with full Balatro mechanics.

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

    # Phase 1: Card scoring — each scoring card adds its chip value + enhancement
    card_chips = 0
    add_chips = 0
    add_mult = 0
    x_mult = 1.0

    for idx in scoring_idxs:
        card = played_cards[idx]
        # Stone cards don't add rank chips, only their +50
        if card.enhancement == "Stone Card":
            card_chips += ENHANCEMENT_CHIPS.get("Stone Card", 50)
        else:
            card_chips += card.chip_value
            card_chips += ENHANCEMENT_CHIPS.get(card.enhancement, 0)

        add_mult += ENHANCEMENT_MULT.get(card.enhancement, 0)

        if card.enhancement in ENHANCEMENT_XMULT:
            x_mult *= ENHANCEMENT_XMULT[card.enhancement]

        # Edition bonuses on scoring cards
        add_chips += EDITION_CHIPS.get(card.edition, 0)
        add_mult += EDITION_MULT.get(card.edition, 0)
        if card.edition in EDITION_XMULT:
            x_mult *= EDITION_XMULT[card.edition]

        # Red Seal retrigger (simplified: double the card's contribution)
        if card.seal in SEAL_RETRIGGER:
            card_chips += card.chip_value
            add_mult += ENHANCEMENT_MULT.get(card.enhancement, 0)

    # Phase 2: Joker effects (simplified — covers common jokers)
    joker_names = {j.name for j in jokers}
    for j in jokers:
        jc, jm, jx = _joker_scoring_effect(j, hand_type, played_cards, scoring_idxs, held_cards, jokers)
        add_chips += jc
        add_mult += jm
        if jx != 1.0:
            x_mult *= jx

        # Joker edition
        add_chips += EDITION_CHIPS.get(j.edition, 0)
        add_mult += EDITION_MULT.get(j.edition, 0)
        if j.edition in EDITION_XMULT:
            x_mult *= EDITION_XMULT[j.edition]

    total_chips = base_chips + card_chips + add_chips
    total_mult = base_mult + add_mult
    final_score = total_chips * total_mult * x_mult

    return ScoreBreakdown(
        hand_type=hand_type,
        hand_rank=hand_rank,
        base_chips=base_chips,
        base_mult=base_mult,
        card_chips=card_chips,
        add_chips=add_chips,
        add_mult=add_mult,
        x_mult=x_mult,
        final_score=final_score,
        scoring_cards=scoring_idxs,
        all_cards=[i for i in range(len(played_cards))],
    )


def _joker_scoring_effect(
    joker: Joker,
    hand_type: str,
    played: list[Card],
    scoring_idxs: list[int],
    held: list[Card] | None,
    all_jokers: list[Joker] | None = None,
) -> tuple[int, int, float]:
    """Return (add_chips, add_mult, x_mult) for a joker's scoring effect.

    This covers the most common jokers. Exotic/conditional jokers
    that need full game state are handled by the LLM advisor.
    """
    name = joker.name
    chips, mult, xm = 0, 0, 1.0

    # --- Flat chip jokers ---
    if name == "Joker":
        mult += 4
    elif name == "Green Joker":
        # +1 mult per hand played this round, max +5. Estimate avg +3
        mult += 3
    elif name == "Blue Joker":
        # +2 chips per remaining card in deck. Estimate ~30 cards left
        chips += 60
    elif name == "Red Card":
        # +3 mult per booster pack skipped. Estimate +3
        mult += 3
    elif name == "Ceremonial Dagger":
        # When blind is selected, destroy right-most joker and add double its sell value as mult
        pass  # complex, skip
    elif name == "Mime":
        # Retrigger all held-in-hand effects
        pass  # complex
    elif name == "Acrobat":
        # x3 mult on final hand of round
        pass  # needs hand count context
    elif name == "Sock and Buskin":
        # Retrigger all face cards
        pass  # complex
    elif name == "Swashbuckler":
        # +mult equal to total sell value of owned jokers. Estimate +8
        mult += 8
    elif name == "Troubadour":
        # +2 hand size, -1 hand per round
        pass
    elif name == "Hanging Chad":
        # Retrigger first scoring card 2 additional times
        pass  # complex
    elif name == "Rough Gem":
        # +$1 per Diamond card scored
        pass  # economy
    elif name == "Bloodstone":
        # 1 in 2 chance for x1.5 mult per Heart scored. Estimate avg
        heart_count = sum(1 for i in scoring_idxs if played[i].suit == "Hearts")
        if heart_count > 0:
            xm *= (1.0 + 0.25 * heart_count)  # ~50% chance of x1.5 per heart
    elif name == "Arrowhead":
        chips += 50 * sum(1 for i in scoring_idxs if played[i].suit == "Spades")
    elif name == "Onyx Agate":
        mult += 7 * sum(1 for i in scoring_idxs if played[i].suit == "Clubs")
    elif name == "Greedy Joker":
        chips += 3 * sum(1 for i in scoring_idxs if played[i].suit == "Diamonds")
    elif name == "Lusty Joker":
        mult += 3 * sum(1 for i in scoring_idxs if played[i].suit == "Hearts")
    elif name == "Wrathful Joker":
        mult += 3 * sum(1 for i in scoring_idxs if played[i].suit == "Spades")
    elif name == "Gluttonous Joker":
        mult += 3 * sum(1 for i in scoring_idxs if played[i].suit == "Clubs")
    elif name == "Jolly Joker":
        if hand_type in ("Pair", "Two Pair", "Full House", "Four of a Kind", "Five of a Kind"):
            mult += 8
    elif name == "Zany Joker":
        if hand_type in ("Three of a Kind", "Full House", "Four of a Kind", "Five of a Kind"):
            mult += 12
    elif name == "Mad Joker":
        if hand_type in ("Two Pair", "Full House"):
            mult += 10
    elif name == "Crazy Joker":
        if hand_type in ("Straight", "Straight Flush"):
            mult += 12
    elif name == "Droll Joker":
        if hand_type in ("Flush", "Flush House", "Flush Five", "Straight Flush"):
            mult += 10
    elif name == "Sly Joker":
        if hand_type in ("Pair", "Two Pair", "Full House", "Four of a Kind"):
            chips += 50
    elif name == "Wily Joker":
        if hand_type in ("Three of a Kind", "Full House", "Four of a Kind"):
            chips += 100
    elif name == "Clever Joker":
        if hand_type in ("Two Pair", "Full House"):
            chips += 80
    elif name == "Devious Joker":
        if hand_type in ("Straight", "Straight Flush"):
            chips += 100
    elif name == "Crafty Joker":
        if hand_type in ("Flush", "Flush House", "Flush Five", "Straight Flush"):
            chips += 80
    elif name == "Half Joker":
        if len(played) <= 3:
            mult += 20
    elif name == "Stencil Joker":
        # +1 xMult per empty joker slot (assume 5 slots)
        joker_count = len(all_jokers) if all_jokers else 1
        empty = max(0, 5 - joker_count)
        if empty > 0:
            xm *= (1.0 + empty)
    elif name == "Misprint":
        mult += 12  # average of 0-23
    elif name == "Raised Fist":
        if held:
            lowest = min(held, key=lambda c: c.rank_num)
            mult += lowest.rank_num
    elif name == "Banner":
        mult += 30  # +30 chips per discard remaining — estimate 1 discard left
    elif name == "Mystic Summit":
        mult += 8  # +15 mult if 0 discards left — estimate 50% chance
    elif name == "Loyalty Card":
        xm *= 1.2  # xMult every 6 hands — estimate average contribution
    elif name == "Scary Face":
        chips += 30 * sum(1 for i in scoring_idxs
                          if played[i].rank in ("Jack", "Queen", "King"))
    elif name == "Abstract Joker":
        joker_count = len(all_jokers) if all_jokers else 1
        mult += 3 * joker_count
    elif name == "Ride the Bus":
        mult += 3  # +1 mult per consecutive non-face hand — estimate avg +3
    elif name == "Supernova":
        mult += 3  # +mult equal to times this hand type was played — estimate avg +3
    elif name == "Blackboard":
        if held and all(c.suit in ("Spades", "Clubs") for c in held):
            xm *= 3.0
    elif name == "The Duo":
        if hand_type in ("Pair", "Two Pair", "Full House", "Four of a Kind", "Five of a Kind"):
            xm *= 2.0
    elif name == "The Trio":
        if hand_type in ("Three of a Kind", "Full House", "Four of a Kind", "Five of a Kind"):
            xm *= 3.0
    elif name == "The Family":
        if hand_type in ("Four of a Kind", "Five of a Kind"):
            xm *= 4.0
    elif name == "The Order":
        if hand_type in ("Straight", "Straight Flush"):
            xm *= 3.0
    elif name == "The Tribe":
        if hand_type in ("Flush", "Flush House", "Flush Five", "Straight Flush"):
            xm *= 2.0
    elif name == "Fibonacci":
        fib_ranks = {"Ace", "2", "3", "5", "8"}
        mult += 8 * sum(1 for i in scoring_idxs if played[i].rank in fib_ranks)
    elif name == "Steel Joker":
        if held:
            steel_count = sum(1 for c in held if c.enhancement == "Steel Card")
            if steel_count:
                xm *= (1.0 + 0.2 * steel_count)
    elif name == "Photograph":
        # First face card played gets xMult
        for i in scoring_idxs:
            if played[i].rank in ("Jack", "Queen", "King"):
                xm *= 2.0
                break
    elif name == "Even Steven":
        even_ranks = {"2", "4", "6", "8", "10"}
        mult += 4 * sum(1 for i in scoring_idxs if played[i].rank in even_ranks)
    elif name == "Odd Todd":
        odd_ranks = {"3", "5", "7", "9", "Ace"}
        chips += 31 * sum(1 for i in scoring_idxs if played[i].rank in odd_ranks)
    elif name == "Scholar":
        chips += 20 * sum(1 for i in scoring_idxs if played[i].rank == "Ace")
        mult += 4 * sum(1 for i in scoring_idxs if played[i].rank == "Ace")
    elif name == "Walkie Talkie":
        count_10_4 = sum(1 for i in scoring_idxs if played[i].rank in ("10", "4"))
        chips += 10 * count_10_4
        mult += 4 * count_10_4
    elif name == "Supernova":
        pass  # already handled above
    elif name == "Hiker":
        # Permanently +5 chips to each played card — estimate avg +15
        chips += 15

    return chips, mult, xm


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
