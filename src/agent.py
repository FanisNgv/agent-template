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

SECONDARY GOALS:
- NWA (Nash Welfare Above BATNA): both players should gain well ABOVE their BATNA.
- EF1 (Envy-Freeness up to 1 item): give opponent enough that they don't envy you
  after removing your single most valuable item. This means give opponent items they
  value highly, even if you value them too.

STRATEGY GUIDELINES:
- Propose splits where BOTH parties gain well above their BATNA.
- Assume the opponent values items differently from you — give them items you value less.
- A fair deal that both sides accept beats a greedy proposal that gets rejected.
- Concede more as rounds run out — a deal at discount is better than no deal.
- Accept when the offer is reasonable; walking away only gives you BATNA.
- Target roughly 50% of the total possible value — not 60%, not 70%. Greedy = no deal.

Respond with valid JSON ONLY — no prose, no markdown fences."""


def _extract_json(text: str) -> str:
    """Strip markdown fences if present and return raw JSON string."""
    text = text.strip()
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if match:
        return match.group(1).strip()
    return text


def _infer_opponent_value_weights(obs: dict) -> list[float]:
    """Return estimated per-unit value weights for the opponent (unnormalized).

    Based on what fraction of each item the opponent kept for themselves across
    all observed proposals. Returns uniform weights if no history is available.
    """
    quantities: list[int] = obs.get("quantities", [])
    n = len(quantities)
    if n == 0:
        return []

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
        return [1.0] * n

    avg_kept_frac = []
    for i in range(n):
        if quantities[i] > 0:
            avg = sum(h[i] for h in opponent_kept_history) / (len(opponent_kept_history) * quantities[i])
        else:
            avg = 0.0
        avg_kept_frac.append(avg)

    # Add small epsilon so we never assign zero weight to any item
    return [max(0.05, f) for f in avg_kept_frac]


def _proportional_proposal(obs: dict) -> dict:
    """Proportional-fairness proposal anchored on Nash Bargaining Solution.

    Allocates each item proportionally to relative per-unit values:
      self_share[i] = v_self[i] / (v_self[i] + v_opp_inferred[i])

    This tends to maximise Nash Welfare AND achieve EF1 fairness.
    A slight self-advantage is applied early (advantage > 1) and fades under
    time pressure so we concede as rounds run out.
    """
    quantities = obs["quantities"]
    valuations = obs["valuations_self"]
    batna_self = obs.get("batna_self", 0)
    round_index = obs.get("round_index", 1)
    max_rounds = obs.get("max_rounds", 5)
    n = len(quantities)

    opp_weights = _infer_opponent_value_weights(obs)

    # Scale opponent weights so they are comparable per-unit values to ours.
    # Strategy: normalise so that the maximum opponent weight equals our maximum
    # per-unit valuation.  With uniform weights (no history) this gives a ~50/50
    # split, which is the neutral/generous starting point.  With actual history
    # signals the relative ranking of weights drives allocation correctly.
    max_val = max(valuations) if valuations else 1
    max_opp_w = max(opp_weights) if opp_weights else 1.0
    scale_factor = max_val / max_opp_w
    opp_vals = [w * scale_factor for w in opp_weights]

    # Self-advantage factor: start slightly above 1, decline under time pressure.
    # 1.10 early → 0.85 in the final round, so we concede meaningfully.
    time_pressure = round_index / max_rounds
    advantage = 1.10 - 0.30 * time_pressure  # 1.10 → 0.80 over rounds

    alloc_self = []
    alloc_other = []

    for i in range(n):
        q = quantities[i]
        v_self = valuations[i] * advantage
        v_opp = opp_vals[i]

        if v_self + v_opp > 0:
            self_frac = v_self / (v_self + v_opp)
        else:
            self_frac = 0.5

        self_qty = round(self_frac * q)
        # Always leave at least 1 unit of every item for opponent (helps EF1)
        self_qty = max(0, min(q - 1 if q > 0 else 0, self_qty))

        alloc_self.append(self_qty)
        alloc_other.append(q - self_qty)

    # If we'd fall below BATNA, greedily take more of our highest-value items
    self_utility = sum(valuations[i] * alloc_self[i] for i in range(n))
    if self_utility < batna_self * 1.05 and batna_self > 0:
        order = sorted(range(n), key=lambda i: valuations[i], reverse=True)
        for idx in order:
            if self_utility >= batna_self * 1.05:
                break
            can_take = alloc_other[idx] - 1  # keep at least 1 for opponent
            if can_take > 0:
                per_unit = max(valuations[idx], 1)
                need = int((batna_self * 1.05 - self_utility) / per_unit) + 1
                take = min(can_take, need)
                alloc_self[idx] += take
                alloc_other[idx] -= take
                self_utility += take * valuations[idx]

    return {"allocation_self": alloc_self, "allocation_other": alloc_other}


def _infer_opponent_preferences(obs: dict) -> str:
    """Format opponent preference signal as a human-readable string for the LLM prompt."""
    quantities: list[int] = obs.get("quantities", [])
    n = len(quantities)
    if n == 0:
        return ""

    opp_weights = _infer_opponent_value_weights(obs)
    if all(w == 1.0 for w in opp_weights):
        return ""  # No history — no signal

    ranked = sorted(range(n), key=lambda i: opp_weights[i], reverse=True)
    total = sum(opp_weights)
    lines = [f"  Item {i}: relative preference score {opp_weights[i]/total*100:.0f}%" for i in ranked]
    top = [str(i) for i in ranked if opp_weights[i] / total > 0.4]
    bottom = [str(i) for i in ranked if opp_weights[i] / total < 0.15]

    summary = "\n".join(lines)
    insight = ""
    if top:
        insight += f"\n  → Opponent strongly values item(s) {', '.join(top)} — give these to them."
    if bottom:
        insight += f"\n  → Opponent barely values item(s) {', '.join(bottom)} — keep those for yourself."

    n_obs = sum(1 for _ in obs.get("history", [])) + (1 if obs.get("last_offer") else 0)
    return f"OPPONENT PREFERENCE SIGNALS (from {n_obs} observed proposal(s)):\n{summary}{insight}"


def _repair_proposal(result: dict, quantities: list[int]) -> dict:
    """Clamp allocation_self to valid range and recompute allocation_other."""
    n = len(quantities)
    alloc_self = result.get("allocation_self", [0] * n)
    alloc_self = [max(0, min(int(alloc_self[i]), quantities[i])) for i in range(n)]
    alloc_other = [quantities[i] - alloc_self[i] for i in range(n)]
    return {"allocation_self": alloc_self, "allocation_other": alloc_other}


def _accept_or_reject_rules(obs: dict) -> dict:
    """Pure deterministic accept/reject rules — no LLM needed.

    Implements Rubinstein subgame perfect equilibrium logic combined with
    practical concession thresholds to maximise deal rate and NWA.
    """
    offer_value: float = obs.get("offer_value", 0)
    batna_value: float = obs.get("batna_value", obs.get("batna_self", 0))
    counter_value: float = obs.get("counter_value", 0)
    round_index: int = obs.get("round_index", 1)
    max_rounds: int = obs.get("max_rounds", 5)
    discount: float = obs.get("discount", 0.98)

    # Rule 1: Never accept below BATNA
    if offer_value < batna_value:
        return {"accept": False}

    rounds_left = max_rounds - round_index

    # Rule 2: Last round — always accept (no more chances)
    if rounds_left == 0:
        return {"accept": True}

    # Rule 3: Rubinstein optimal — accept if offer >= discounted counter-offer.
    # Waiting one more round costs a factor of `discount`, so:
    #   accept if offer_value >= counter_value * discount
    discounted_counter = counter_value * discount
    if offer_value >= discounted_counter:
        return {"accept": True}

    # Rule 4: Any offer >= 25% above BATNA is a good deal — accept immediately.
    # Holding out for marginal improvements risks no-deal (NWA = 0).
    if offer_value >= batna_value * 1.25:
        return {"accept": True}

    # Rule 5: Mid-game (≥ 40% of rounds elapsed) — accept if meaningfully above BATNA
    mid_game = round_index >= max_rounds * 0.4
    if mid_game and offer_value >= batna_value * 1.15:
        return {"accept": True}

    # Rule 6: Late game (≥ 60% of rounds elapsed) — accept at 10% above BATNA
    late_game = round_index >= max_rounds * 0.6
    if late_game and offer_value >= batna_value * 1.10:
        return {"accept": True}

    # Rule 7: Very late game (≥ 80%) — accept anything above BATNA
    very_late = round_index >= max_rounds * 0.8
    if very_late and offer_value >= batna_value:
        return {"accept": True}

    # Otherwise reject — counter-offer should be better
    return {"accept": False}


class Agent:
    def __init__(self):
        self.messenger = Messenger()
        self._client: AsyncOpenAI | None = None
        self.model = "gpt-4o"

    @property
    def client(self) -> AsyncOpenAI:
        """Lazily initialise the OpenAI client on first use."""
        if self._client is None:
            self._client = AsyncOpenAI(
                api_key=os.environ.get("OPENAI_API_KEY"),
                timeout=30.0,
            )
        return self._client

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
                result = _accept_or_reject_rules(obs)
            else:
                result = await self._decide_proposal(obs)
        except Exception as e:
            print(f"LLM error, using heuristic fallback: {e}")
            if action == "ACCEPT_OR_REJECT":
                result = _accept_or_reject_rules(obs)
            else:
                result = _proportional_proposal(obs)

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

        # Compute analytical anchor proposal to show the LLM a concrete reference
        anchor = _proportional_proposal(obs)
        anchor_self_value = sum(valuations[i] * anchor["allocation_self"][i] for i in range(n))
        anchor_pct = (anchor_self_value / total_value * 100) if total_value > 0 else 0

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
Your minimum required value (BATNA): {batna}

{opponent_signals}

ANALYTICAL ANCHOR (proportional-fairness baseline):
  allocation_self:  {anchor["allocation_self"]}
  allocation_other: {anchor["allocation_other"]}
  → gives you {anchor_self_value:.0f} ({anchor_pct:.0f}% of your maximum)

TASK: Propose an allocation that maximises Nash Welfare = your_utility × opponent_utility.

Think step by step:
1. Keep items YOU value most (high valuation per unit).
2. Use opponent signals: give them items they value highly.
3. TARGET: your value ≈ 50% of your maximum (not 60%+; greedy = rejected = no deal).
4. EF1 goal: give opponent at least 1 unit of every item so they don't envy you entirely.
5. NWA goal: ensure BOTH sides are well above their BATNA — this is what NWA measures.
6. If rounds are running out, concede more toward the analytical anchor above.
7. The anchor is a good starting point; you may adjust if opponent signals suggest a better split.

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
        repaired = _repair_proposal(result, quantities)

        # Sanity check: if LLM proposal gives us less than BATNA, fall back to analytical
        llm_self_value = sum(valuations[i] * repaired["allocation_self"][i] for i in range(n))
        if llm_self_value < batna * 0.95:
            print(f"LLM proposal below BATNA ({llm_self_value:.1f} < {batna}), using analytical fallback")
            return _proportional_proposal(obs)

        return repaired
