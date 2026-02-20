# Balatro AI Strategy

Each branch represents a strategy version. The `main` branch has the latest.

## Structure
- `engine.py` — Decision engine (Rule vs LLM routing)
- `strategy.py` — Game context, shop evaluation, archetype tracking
- `scoring.py` — Hand evaluation, score estimation, joker effects
- `llm_advisor.py` — LLM system prompt and JSON parsing
- `strategy.json` — Strategy metadata (name, model, params)

## Strategy Evolution
New strategies are created as branches from their parent.
