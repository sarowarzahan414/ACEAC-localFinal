"""
Baseline Agents for ACEAC Validation

These are simple agents that provide reality checks:
- RandomAgent: Sanity check (should lose to any trained agent)
- GreedyAgent: Simple heuristic (should be beatable but non-trivial)
- PassiveAgent: Does nothing (should lose decisively)

If your trained agents can't beat these baselines significantly,
your training doesn't work.
"""

import numpy as np
from typing import Tuple


class RandomAgent:
    """
    Random action selection baseline

    This is the MINIMUM bar. Any trained agent should beat this >70% win rate.
    If your agent can't beat random, training failed completely.
    """

    def __init__(self, num_actions: int = 25):
        self.num_actions = num_actions
        self.name = "Random"

    def predict(self, obs, deterministic: bool = True) -> Tuple[int, None]:
        """Random action selection"""
        return np.random.randint(0, self.num_actions), None


class GreedyAgent:
    """
    Simple state-based heuristic

    Strategy:
    - Red: Pick actions that favor high state values
    - Blue: Pick actions that favor low state values
    - Action selection based on current state position

    This is slightly smarter than random but still simplistic.
    Trained agents should beat this >60% win rate to show meaningful learning.
    """

    def __init__(self, role: str = "red", num_actions: int = 25):
        self.role = role
        self.num_actions = num_actions
        self.name = f"Greedy-{role}"

    def predict(self, obs, deterministic: bool = True) -> Tuple[int, None]:
        """State-based action selection"""
        state_mean = np.mean(obs)

        if self.role == "red":
            # Red wants high state
            # If state is low, be very aggressive (high action numbers)
            # If state is high, maintain (medium action numbers)
            if state_mean < 0.3:
                # Desperate attack
                action = np.random.randint(15, 25)
            elif state_mean < 0.5:
                # Aggressive attack
                action = np.random.randint(10, 20)
            elif state_mean < 0.7:
                # Moderate attack
                action = np.random.randint(5, 15)
            else:
                # Maintain/consolidate
                action = np.random.randint(0, 10)
        else:
            # Blue wants low state
            # If state is high, be very aggressive defensive (high action numbers)
            # If state is low, maintain (medium action numbers)
            if state_mean > 0.7:
                # Desperate defense
                action = np.random.randint(15, 25)
            elif state_mean > 0.5:
                # Aggressive defense
                action = np.random.randint(10, 20)
            elif state_mean > 0.3:
                # Moderate defense
                action = np.random.randint(5, 15)
            else:
                # Maintain/consolidate
                action = np.random.randint(0, 10)

        return action, None


class PassiveAgent:
    """
    Does minimal actions

    This should ALWAYS lose. If it doesn't, something is broken.
    """

    def __init__(self):
        self.name = "Passive"

    def predict(self, obs, deterministic: bool = True) -> Tuple[int, None]:
        """Always action 0 (minimal effect)"""
        return 0, None


class AlwaysAggressiveAgent:
    """
    Always picks maximum action (action 24)

    Simple strategy: maximum aggression always.
    Trained agents should beat this by being more adaptive.
    """

    def __init__(self):
        self.name = "AlwaysAggressive"

    def predict(self, obs, deterministic: bool = True) -> Tuple[int, None]:
        """Always maximum action"""
        return 24, None


def get_baseline_agent(baseline_type: str, role: str = "red"):
    """
    Factory function to get baseline agents

    Args:
        baseline_type: "random", "greedy", "passive", "aggressive"
        role: "red" or "blue" (for greedy agent)

    Returns:
        Baseline agent instance
    """
    if baseline_type.lower() == "random":
        return RandomAgent()
    elif baseline_type.lower() == "greedy":
        return GreedyAgent(role=role)
    elif baseline_type.lower() == "passive":
        return PassiveAgent()
    elif baseline_type.lower() == "aggressive":
        return AlwaysAggressiveAgent()
    else:
        raise ValueError(f"Unknown baseline type: {baseline_type}")


if __name__ == "__main__":
    # Test baseline agents
    print("Testing baseline agents...")

    obs = np.array([0.5] * 32, dtype=np.float32)

    agents = [
        RandomAgent(),
        GreedyAgent(role="red"),
        GreedyAgent(role="blue"),
        PassiveAgent(),
        AlwaysAggressiveAgent()
    ]

    for agent in agents:
        action, _ = agent.predict(obs)
        print(f"{agent.name:20s} -> Action: {action}")

    print("\n✓ Baseline agents working")
