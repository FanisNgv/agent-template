"""Quick local smoke test — runs agent logic directly without the A2A server."""
import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

from a2a.types import Message, Part, Role, TextPart
from a2a.server.tasks import TaskUpdater

import sys
sys.path.insert(0, "src")
from agent import Agent


def make_message(obs: dict) -> Message:
    return Message(
        kind="message",
        role=Role.user,
        parts=[Part(root=TextPart(text=json.dumps(obs)))],
        message_id="test-001",
    )


def mock_updater():
    updater = AsyncMock(spec=TaskUpdater)
    updater._terminal_state_reached = False
    return updater


async def test_propose():
    print("\n--- PROPOSE ---")
    obs = {
        "action": "propose",
        "role": "row",
        "valuations_self": [80, 40, 10],
        "batna_self": 50,
        "discount": 0.98,
        "max_rounds": 5,
        "quantities": [7, 4, 1],
        "round_index": 1,
    }
    agent = Agent()
    updater = mock_updater()
    await agent.run(make_message(obs), updater)

    call = updater.add_artifact.call_args
    text = call.kwargs["parts"][0].root.text
    result = json.loads(text)
    print(f"Observation: valuations={obs['valuations_self']}, batna={obs['batna_self']}, round=1/5")
    print(f"Response:    {result}")
    assert "allocation_self" in result
    assert "allocation_other" in result
    for i in range(3):
        assert result["allocation_self"][i] + result["allocation_other"][i] == obs["quantities"][i], \
            f"Constraint violated for item {i}"
    print("✓ Constraints satisfied")


async def test_accept():
    print("\n--- ACCEPT (offer > BATNA) ---")
    obs = {
        "action": "ACCEPT_OR_REJECT",
        "role": "col",
        "valuations_self": [80, 40, 10],
        "batna_self": 50,
        "batna_value": 50,
        "offer_value": 180,
        "counter_value": 200,
        "discount": 0.98,
        "max_rounds": 5,
        "quantities": [7, 4, 1],
        "round_index": 4,
    }
    agent = Agent()
    updater = mock_updater()
    await agent.run(make_message(obs), updater)

    call = updater.add_artifact.call_args
    result = json.loads(call.kwargs["parts"][0].root.text)
    print(f"offer_value={obs['offer_value']}, batna={obs['batna_value']}, round=4/5")
    print(f"Response: {result}")
    assert "accept" in result
    print(f"✓ Decision: {'ACCEPT' if result['accept'] else 'REJECT'}")


async def test_reject_below_batna():
    print("\n--- REJECT (offer < BATNA, hard rule) ---")
    obs = {
        "action": "ACCEPT_OR_REJECT",
        "role": "row",
        "valuations_self": [80, 40, 10],
        "batna_self": 100,
        "batna_value": 100,
        "offer_value": 40,
        "counter_value": 150,
        "discount": 0.98,
        "max_rounds": 5,
        "quantities": [7, 4, 1],
        "round_index": 2,
    }
    agent = Agent()
    updater = mock_updater()
    await agent.run(make_message(obs), updater)

    call = updater.add_artifact.call_args
    result = json.loads(call.kwargs["parts"][0].root.text)
    print(f"offer_value={obs['offer_value']}, batna={obs['batna_value']}")
    print(f"Response: {result}")
    assert result["accept"] is False, "Must reject when offer < BATNA"
    print("✓ Correctly rejected (below BATNA)")


async def main():
    print("=== Local agent smoke test ===")
    await test_propose()
    await test_accept()
    await test_reject_below_batna()
    print("\n=== All tests passed ===")


if __name__ == "__main__":
    asyncio.run(main())
