import asyncio
import json
import os
import re
import math

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

def _estimate_opponent_values(obs: dict) -> list[float]:
    """Estimate opponent per-unit values from revealed behavior.

    If no history is available, assume inverse correlation with our values.
    """
    quantities: list[int] = obs.get("quantities", [])
    self_vals: list[int] = obs.get("valuations_self", [])
    n = len(quantities)
    if n == 0:
        return []

    # Base prior: inverse of our values (opponent likes what we like less).
    max_self = max(self_vals) if self_vals else 1
    base = [max_self + 1 - v for v in self_vals]
    base_avg = sum(base) / n if n else 1.0
    base_norm = [v / base_avg if base_avg > 0 else 1.0 for v in base]

    # Collect opponent kept fractions from history/last_offer.
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

    last_offer = obs.get("last_offer")
    if last_offer and len(last_offer) == n:
        kept = [quantities[i] - int(last_offer[i]) for i in range(n)]
        if all(k >= 0 for k in kept):
            opponent_kept_history.append(kept)

    if not opponent_kept_history:
        # Scale prior to roughly match our value scale.
        self_avg = sum(self_vals) / n if n else 1.0
        scale = self_avg / (sum(base_norm) / n) if sum(base_norm) > 0 else 1.0
        return [max(0.1, v * scale) for v in base_norm]

    k = len(opponent_kept_history)
    kept_frac = []
    for i in range(n):
        if quantities[i] > 0:
            avg = sum(h[i] for h in opponent_kept_history) / (k * quantities[i])
        else:
            avg = 0.0
        kept_frac.append(avg)

    # Map kept fraction to a multiplier in [0.5, 2.0]
    multipliers = [0.5 + 1.5 * min(1.0, max(0.0, f)) for f in kept_frac]

    # Combine prior + signals and scale to our average value.
    opp_vals = [base_norm[i] * multipliers[i] for i in range(n)]
    opp_avg = sum(opp_vals) / n if n else 1.0
    self_avg = sum(self_vals) / n if n else 1.0
    scale = self_avg / opp_avg if opp_avg > 0 else 1.0
    return [max(0.1, v * scale) for v in opp_vals]


def _infer_opponent_preferences(obs: dict) -> str:
    """Infer opponent's item preferences from their past proposals.

    The opponent reveals preferences by what they keep for themselves.
    We observe this via last_offer (what they offered US) and history.
    """
    quantities: list[int] = obs.get("quantities", [])
    n = len(quantities)
    if n == 0:
        return ""

    # Collect all opponent proposals (what they kept = quantities - offer_to_us)
    opponent_kept_history: list[list[int]] = []

    # From history field
    history = obs.get("history", [])
    for entry in history:
        # history entries may be dicts with an "offer" key, or plain lists
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

    # From last_offer (most recent opponent proposal to us)
    last_offer = obs.get("last_offer")
    if last_offer and len(last_offer) == n:
        kept = [quantities[i] - int(last_offer[i]) for i in range(n)]
        if all(k >= 0 for k in kept):
            opponent_kept_history.append(kept)

    if not opponent_kept_history:
        return ""

    # Average fraction of each item the opponent kept for themselves
    avg_kept_frac = []
    for i in range(n):
        if quantities[i] > 0:
            avg = sum(h[i] for h in opponent_kept_history) / (len(opponent_kept_history) * quantities[i])
        else:
            avg = 0.0
        avg_kept_frac.append(avg)

    # Rank items by how much opponent wants them
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

    return f"OPPONENT PREFERENCE SIGNALS (from {len(opponent_kept_history)} observed proposal(s)):\n{summary}{insight}"


def _repair_proposal(result: dict, quantities: list[int]) -> dict:
    """Clamp allocation_self to valid range and recompute allocation_other.

    Enforce opponent gets at least 1 unit where possible (helps EF1).
    """
    n = len(quantities)
    alloc_self = result.get("allocation_self", [0] * n)
    alloc_self = [
        max(0, min(int(alloc_self[i]), max(0, quantities[i] - (1 if quantities[i] > 0 else 0))))
        for i in range(n)
    ]
    alloc_other = [quantities[i] - alloc_self[i] for i in range(n)]
    return {"allocation_self": alloc_self, "allocation_other": alloc_other}

def _target_self_fraction(round_index: int, max_rounds: int) -> float:
    progress = round_index / max_rounds if max_rounds else 1.0
    if progress < 0.4:
        return 0.60
    if progress < 0.7:
        return 0.55
    return 0.50

def _optimize_nash_allocation(
    quantities: list[int],
    self_vals: list[int],
    opp_vals: list[float],
    batna: float,
    round_index: int,
    max_rounds: int,
) -> dict:
    """Greedy unit allocation to maximize Nash product with fairness constraints."""
    n = len(quantities)
    opp_min = [1 if q > 0 else 0 for q in quantities]

    alloc_self = [0] * n
    alloc_other = opp_min[:]

    self_value = 0.0
    opp_value = sum(opp_vals[i] * alloc_other[i] for i in range(n))

    remaining = [quantities[i] - alloc_other[i] for i in range(n)]

    # Greedy allocation of remaining units to maximize log Nash product.
    eps = 1e-6
    total_remaining = sum(remaining)
    for _ in range(total_remaining):
        best_i = None
        best_to_self = True
        best_delta = -1e9
        s = max(self_value, eps)
        o = max(opp_value, eps)

        for i in range(n):
            if remaining[i] <= 0:
                continue
            delta_self = math.log((s + self_vals[i]) / s)
            delta_opp = math.log((o + opp_vals[i]) / o)
            if delta_self >= delta_opp:
                delta = delta_self
                to_self = True
            else:
                delta = delta_opp
                to_self = False
            if delta > best_delta:
                best_delta = delta
                best_i = i
                best_to_self = to_self

        if best_i is None:
            break

        if best_to_self:
            alloc_self[best_i] += 1
            self_value += self_vals[best_i]
        else:
            alloc_other[best_i] += 1
            opp_value += opp_vals[best_i]
        remaining[best_i] -= 1

    # Enforce a minimum self value target (time-aware) + BATNA.
    max_self_value = sum(self_vals[i] * quantities[i] for i in range(n))
    target_fraction = _target_self_fraction(round_index, max_rounds)
    target_self = max(batna, target_fraction * max_self_value)

    if self_value < target_self:
        # Move units from opponent to self with best gain/impact ratio.
        while self_value < target_self:
            best_i = None
            best_ratio = -1.0
            for i in range(n):
                if alloc_other[i] > opp_min[i]:
                    ratio = self_vals[i] / max(opp_vals[i], eps)
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_i = i
            if best_i is None:
                break
            alloc_other[best_i] -= 1
            alloc_self[best_i] += 1
            self_value += self_vals[best_i]
            opp_value -= opp_vals[best_i]

    return {"allocation_self": alloc_self, "allocation_other": alloc_other}


class Agent:
    def __init__(self):
        self.messenger = Messenger()
        self.client = AsyncOpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
            timeout=30.0,
        )
        # Stronger than mini baselines but not top-tier cost.
        self.model = "gpt-5-mini"
        # Minimize LLM involvement by default for stability; opt-in via env.
        self.use_llm = os.environ.get("USE_LLM", "0") == "1"

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

        if not self.use_llm:
            opp_vals = _estimate_opponent_values(obs)
            result = _optimize_nash_allocation(
                quantities=quantities,
                self_vals=valuations,
                opp_vals=opp_vals,
                batna=batna,
                round_index=round_index,
                max_rounds=max_rounds,
            )
            return _repair_proposal(result, quantities)

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

Constraint: allocation_self[i] + allocation_other[i] == quantities[i] for every i.
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
        counter_value: float = obs.get("counter_value", 0)
        round_index: int = obs.get("round_index", 1)
        max_rounds: int = obs.get("max_rounds", 5)
        discount: float = obs.get("discount", 0.98)

        # Hard rule: never accept below BATNA
        if offer_value < batna_value:
            return {"accept": False}

        rounds_left = max_rounds - round_index

        # Hard rule: last round — always accept if above BATNA (no more chances)
        if rounds_left == 0:
            return {"accept": True}

        # Late-game bias: accept modest gains above BATNA.
        late_game = round_index >= max_rounds * 0.6
        if late_game and offer_value >= batna_value * 1.05:
            return {"accept": True}

        # Estimate value of countering (discounted next-round expectation).
        next_discount = discount ** round_index
        if counter_value <= 0:
            # Fallback estimate: target a time-aware fraction of max value.
            quantities: list[int] = obs.get("quantities", [])
            valuations: list[int] = obs.get("valuations_self", [])
            max_value = sum(valuations[i] * quantities[i] for i in range(len(quantities)))
            target_fraction = _target_self_fraction(round_index, max_rounds)
            counter_value = max_value * target_fraction
        discounted_counter = counter_value * next_discount

        # Accept if current offer is competitive with expected counter.
        progress = round_index / max_rounds if max_rounds else 1.0
        if progress < 0.4:
            threshold = discounted_counter * 0.95
        elif progress < 0.7:
            threshold = discounted_counter * 0.85
        else:
            threshold = discounted_counter * 0.70

        threshold = max(threshold, batna_value)
        return {"accept": offer_value >= threshold}
