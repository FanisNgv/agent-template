import asyncio
import json
import os
import re

from openai import AsyncOpenAI, RateLimitError
from a2a.server.tasks import TaskUpdater
from a2a.types import Message, TaskState, Part, TextPart
from a2a.utils import get_message_text, new_agent_text_message

from messenger import Messenger

SYSTEM_PROMPT = """\
You are a strategic negotiator in a multi-round bilateral bargaining game.

GAME RULES:
- You and an opponent divide items. Each item has a fixed total quantity.
- You have PRIVATE valuations for each item (opponent does not know yours).
- Every round of delay costs you: payoff = value × discount^(round_index - 1).
- If no deal is reached, both players receive their BATNA (outside option).

YOUR PRIMARY GOAL: maximise Nash Welfare = your_utility × opponent_utility.
This means finding deals that are GOOD FOR BOTH SIDES, not just yourself.

STRATEGY GUIDELINES:
- Propose splits where BOTH parties gain well above their BATNA.
- Assume the opponent values items differently from you — give them items you value less.
- A fair deal that both sides accept beats a greedy proposal that gets rejected.
- Concede more as rounds run out — a deal at discount is better than no deal.
- Accept when the offer is reasonable; walking away only gives you BATNA.

Respond with valid JSON ONLY — no prose, no markdown fences."""


def _extract_json(text: str) -> str:
    """Strip markdown fences if present and return raw JSON string."""
    text = text.strip()
    # Remove ```json ... ``` or ``` ... ```
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if match:
        return match.group(1).strip()
    return text


def _heuristic_proposal(obs: dict) -> dict:
    """Aspiration-style fallback: keep high-value items, concede over time."""
    quantities = obs["quantities"]
    valuations = obs["valuations_self"]
    round_index = obs.get("round_index", 1)
    max_rounds = obs.get("max_rounds", 5)

    time_pressure = round_index / max_rounds
    keep_fraction = 0.65 - 0.15 * time_pressure  # 0.65 → 0.50 over rounds

    total_value = sum(valuations[i] * quantities[i] for i in range(len(quantities)))
    target = keep_fraction * total_value

    # Greedy: keep items sorted by per-unit value
    order = sorted(range(len(quantities)), key=lambda i: valuations[i], reverse=True)
    allocation_self = [0] * len(quantities)
    accumulated = 0.0

    for i in order:
        if accumulated >= target:
            break
        remaining_needed = target - accumulated
        per_unit = valuations[i]
        take = min(quantities[i], max(1, round(remaining_needed / per_unit)))
        # Leave at least 1 unit of each item for opponent (helps EF1)
        take = min(take, max(0, quantities[i] - 1))
        allocation_self[i] = take
        accumulated += take * per_unit

    allocation_other = [quantities[i] - allocation_self[i] for i in range(len(quantities))]
    return {"allocation_self": allocation_self, "allocation_other": allocation_other}


def _infer_opponent_preferences(obs: dict) -> str:
    """Infer opponent's item preferences and concession trend from their past proposals.

    The opponent reveals preferences by what they keep for themselves.
    We observe this via history (chronological) and last_offer.
    """
    quantities: list[int] = obs.get("quantities", [])
    n = len(quantities)
    if n == 0:
        return ""

    # Collect opponent proposals in chronological order (what they kept)
    opponent_kept_history: list[list[int]] = []

    history = obs.get("history", [])
    for entry in history:
        if isinstance(entry, dict):
            offer_to_us = entry.get("offer") or entry.get("allocation_other")
        elif isinstance(entry, list):
            offer_to_us = entry
        else:
            continue
        if offer_to_us and len(offer_to_us) == n:
            kept = [quantities[i] - int(offer_to_us[i]) for i in range(n)]
            if all(k >= 0 for k in kept):
                opponent_kept_history.append(kept)

    # last_offer is most recent
    last_offer = obs.get("last_offer")
    if last_offer and len(last_offer) == n:
        kept = [quantities[i] - int(last_offer[i]) for i in range(n)]
        if all(k >= 0 for k in kept):
            opponent_kept_history.append(kept)

    if not opponent_kept_history:
        return ""

    k = len(opponent_kept_history)

    # Average fraction of each item the opponent kept
    avg_kept_frac = []
    for i in range(n):
        if quantities[i] > 0:
            avg = sum(h[i] for h in opponent_kept_history) / (k * quantities[i])
        else:
            avg = 0.0
        avg_kept_frac.append(avg)

    # Concession trend: compare first half vs second half of history
    trend_lines = []
    if k >= 2:
        mid = k // 2
        early = opponent_kept_history[:mid]
        late = opponent_kept_history[mid:]
        for i in range(n):
            if quantities[i] > 0:
                early_frac = sum(h[i] for h in early) / (len(early) * quantities[i])
                late_frac = sum(h[i] for h in late) / (len(late) * quantities[i])
                delta = late_frac - early_frac
                if abs(delta) > 0.1:
                    direction = "increasing" if delta > 0 else "decreasing"
                    trend_lines.append(f"  Item {i}: opponent demand {direction} ({early_frac*100:.0f}%→{late_frac*100:.0f}%)")

    # Rank items by avg opponent demand
    ranked = sorted(range(n), key=lambda i: avg_kept_frac[i], reverse=True)
    lines = [f"  Item {i}: opponent kept ~{avg_kept_frac[i]*100:.0f}% on average" for i in ranked]
    top = [str(i) for i in ranked if avg_kept_frac[i] > 0.4]
    bottom = [str(i) for i in ranked if avg_kept_frac[i] < 0.2]

    summary = "\n".join(lines)
    insight = ""
    if top:
        insight += f"\n  → Opponent likely values item(s) {', '.join(top)} highly — consider giving these to them."
    if bottom:
        insight += f"\n  → Opponent seems indifferent to item(s) {', '.join(bottom)} — keep those for yourself."
    if trend_lines:
        insight += "\n  TREND:\n" + "\n".join(trend_lines)

    return f"OPPONENT PREFERENCE SIGNALS (from {k} observed proposal(s)):\n{summary}{insight}"


def _repair_proposal(result: dict, quantities: list[int]) -> dict:
    """Clamp allocation_self to valid range, ensure opponent gets ≥1 of each item."""
    n = len(quantities)
    alloc_self = result.get("allocation_self", [0] * n)
    # Clamp to [0, quantities[i]-1] so opponent always gets at least 1 unit (EF1)
    alloc_self = [max(0, min(int(alloc_self[i]), max(0, quantities[i] - 1))) for i in range(n)]
    alloc_other = [quantities[i] - alloc_self[i] for i in range(n)]
    return {"allocation_self": alloc_self, "allocation_other": alloc_other}


class Agent:
    def __init__(self):
        self.messenger = Messenger()
        self.client = AsyncOpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
            timeout=30.0,
        )
        self.model = "gpt-4o"

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        input_text = get_message_text(message)

        await updater.update_status(
            TaskState.working, new_agent_text_message("Analysing negotiation state...")
        )

        try:
            obs = json.loads(input_text)
        except Exception:
            await updater.add_artifact(
                parts=[Part(root=TextPart(text='{"error": "invalid observation JSON"}'))],
                name="negotiation_response",
            )
            return

        action = obs.get("action", "propose")

        try:
            if action == "ACCEPT_OR_REJECT":
                result = await self._decide_accept_reject(obs)
            else:
                result = await self._decide_proposal(obs)
        except Exception as e:
            print(f"LLM error, using heuristic fallback: {e}")
            if action == "ACCEPT_OR_REJECT":
                offer_value = obs.get("offer_value", 0)
                batna_value = obs.get("batna_value", obs.get("batna_self", 0))
                result = {"accept": offer_value >= batna_value}
            else:
                result = _heuristic_proposal(obs)

        await updater.add_artifact(
            parts=[Part(root=TextPart(text=json.dumps(result)))],
            name="negotiation_response",
        )

    # ------------------------------------------------------------------
    # LLM call with retry
    # ------------------------------------------------------------------

    async def _chat(self, messages: list, max_tokens: int = 256) -> str:
        """Call the LLM with up to 3 retries on rate limit errors."""
        for attempt in range(3):
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    messages=messages,
                )
                return response.choices[0].message.content or ""
            except RateLimitError:
                if attempt == 2:
                    raise
                wait = 2 ** attempt  # 1s, 2s
                print(f"Rate limited, retrying in {wait}s...")
                await asyncio.sleep(wait)

    # ------------------------------------------------------------------
    # Proposal
    # ------------------------------------------------------------------

    async def _decide_proposal(self, obs: dict) -> dict:
        quantities: list[int] = obs["quantities"]
        valuations: list[int] = obs["valuations_self"]
        batna: int = obs["batna_self"]
        round_index: int = obs.get("round_index", 1)
        max_rounds: int = obs.get("max_rounds", 5)
        discount: float = obs.get("discount", 0.98)
        role: str = obs.get("role", "row")

        n = len(quantities)
        total_value = sum(valuations[i] * quantities[i] for i in range(n))
        current_discount = discount ** (round_index - 1)
        rounds_left = max_rounds - round_index

        opponent_signals = _infer_opponent_preferences(obs)

        user_prompt = f"""\
NEGOTIATION STATE
-----------------
Items (count): {n}
Available quantities: {quantities}
Your private valuations per unit: {valuations}
Your BATNA (walk-away value): {batna}
Your role: {role}

Round: {round_index} / {max_rounds}  ({rounds_left} rounds remaining after this)
Discount factor: {discount}  →  current multiplier: {current_discount:.4f}

Your maximum possible value (if you kept everything): {total_value}
Your maximum discounted value at this round: {total_value * current_discount:.1f}

{opponent_signals}

TASK: Propose an allocation.

Think step by step (internally):
1. Rank items by your valuation — keep items YOU value most.
2. Use opponent signals above: give them items they value highly (and you value less).
3. Aim for a split where your value is ~60% of your maximum — greedy proposals get rejected.
4. Ensure both sides get well above BATNA to maximise Nash Welfare.
5. Concede more if rounds are running out.

Respond with ONLY this JSON (integers, no extras):
{{"allocation_self": [q0, q1, ...], "allocation_other": [q0, q1, ...]}}

Constraints:
- allocation_self[i] + allocation_other[i] == quantities[i] for every i.
- allocation_other[i] >= 1 for every i (opponent must receive at least 1 unit of each item).
"""

        raw_content = await self._chat(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=256,
        )
        raw = _extract_json(raw_content)
        result = json.loads(raw)
        return _repair_proposal(result, quantities)

    # ------------------------------------------------------------------
    # Accept / Reject
    # ------------------------------------------------------------------

    async def _decide_accept_reject(self, obs: dict) -> dict:
        offer_value: float = obs.get("offer_value", 0)
        batna_value: float = obs.get("batna_value", obs.get("batna_self", 0))
        round_index: int = obs.get("round_index", 1)
        max_rounds: int = obs.get("max_rounds", 5)
        discount: float = obs.get("discount", 0.98)

        # Discounted BATNA: value of waiting shrinks each round
        discounted_batna = batna_value * (discount ** (round_index - 1))

        # Threshold relaxes over time — urgency increases as rounds run out
        progress = round_index / max_rounds
        if progress < 0.33:
            threshold = discounted_batna          # early: require full discounted BATNA
        elif progress < 0.67:
            threshold = discounted_batna * 0.7    # mid: 70%
        else:
            threshold = discounted_batna * 0.3    # late: 30% — strongly prefer closing

        # Hard floor: never accept below actual BATNA (protects NWA%)
        threshold = max(threshold, batna_value)

        return {"accept": offer_value >= threshold}
