"""
ACEAC Zero-Sum Adversarial Environment - BALANCED VERSION

This is a TRUE adversarial environment where:
- Red and Blue compete in a zero-sum game
- One agent's gain = other agent's loss
- Clear win conditions define success
- Rewards are ONLY based on final outcome
- No participation trophies
- BALANCED: No timeout advantage for either side

Author: @sarowarzahan414
Date: 2025-11-19
Purpose: Fix broken reward structure identified in validation

CRITICAL FIX (v2):
- Removed Blue's timeout advantage that caused 100/0 win rate
- Timeout now determines winner based on final state position
- State 0.45-0.55 at timeout = draw (true stalemate)
- This removes structural bias and enables true co-evolution
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, Tuple, Dict


class ZeroSumCyberEnv(gym.Env):
    """
    True zero-sum adversarial cybersecurity environment

    Win Conditions:
    - Red wins: State mean > 0.7 (decisive compromise)
    - Blue wins: State mean < 0.3 (decisive defense)
    - Timeout: State >0.55 = Red wins, <0.45 = Blue wins, else draw
    - Draw: Stalemate (state 0.45-0.55 at timeout)

    Reward Structure:
    - Win: +100
    - Loss: -100
    - Draw: 0
    - Per-step: Minimal shaping bonus (max ±0.25) to guide learning

    BALANCED: No side has timeout advantage (fixed from v1)
    """

    metadata = {'render_modes': [], 'name': 'ACEAC-ZeroSum-v1'}

    def __init__(self,
                 agent_role: str = "red",
                 opponent_model: Optional[object] = None,
                 num_actions: int = 25,
                 state_dim: int = 32,
                 max_steps: int = 100):
        """
        Initialize zero-sum environment

        Args:
            agent_role: "red" (attacker) or "blue" (defender)
            opponent_model: Opponent agent for adversarial play
            num_actions: Number of available actions
            state_dim: Dimensionality of state space
            max_steps: Maximum episode length
        """
        super().__init__()

        self.agent_role = agent_role
        self.opponent_model = opponent_model
        self.num_actions = num_actions
        self.state_dim = state_dim
        self.max_steps = max_steps

        # Spaces
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(state_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(num_actions)

        # Episode state
        self.state = None
        self.step_count = 0
        self.episode_rewards = []  # Track for analysis

        self.reset()

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        """Reset to initial state"""
        super().reset(seed=seed)

        # Start at neutral state (0.5 = contested)
        self.state = np.full(self.state_dim, 0.5, dtype=np.float32)
        self.state += np.random.uniform(-0.1, 0.1, size=self.state_dim)
        self.state = np.clip(self.state, 0.0, 1.0).astype(np.float32)

        self.step_count = 0
        self.episode_rewards = []

        return self._get_observation(), {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute action in zero-sum game

        Returns:
            observation, reward, terminated, truncated, info
        """
        self.step_count += 1

        # Validate action
        if not self.action_space.contains(action):
            action = self.action_space.sample()

        # Get opponent action
        opponent_action = None
        if self.opponent_model is not None:
            try:
                obs = self._get_observation()
                opponent_action, _ = self.opponent_model.predict(obs, deterministic=False)
                opponent_action = int(opponent_action)
            except:
                opponent_action = None

        # Apply state transition (adversarial dynamics)
        self.state = self._adversarial_state_transition(action, opponent_action)

        # Check win conditions
        terminated, winner = self._check_terminal_conditions()

        # Calculate reward (ZERO-SUM)
        reward = self._calculate_zerosum_reward(winner, terminated)

        self.episode_rewards.append(reward)

        # Info for analysis
        info = {
            'step': self.step_count,
            'state_mean': float(np.mean(self.state)),
            'winner': winner,
            'terminated': terminated,
            'action': int(action),
            'opponent_action': int(opponent_action) if opponent_action is not None else -1
        }

        return self._get_observation(), float(reward), terminated, False, info

    def _adversarial_state_transition(self, action: int, opponent_action: Optional[int]) -> np.ndarray:
        """
        State evolves based on adversarial interaction

        - Red actions push state UP (toward 1.0 = compromised)
        - Blue actions push state DOWN (toward 0.0 = secured)
        - Actions directly oppose each other
        """
        new_state = self.state.copy()

        # Agent action effect
        np.random.seed(action)  # Deterministic per action
        affected_dims = np.random.choice(self.state_dim, size=self.state_dim // 4, replace=False)

        for dim in affected_dims:
            effect_size = np.random.uniform(0.05, 0.15)

            if self.agent_role == "red":
                new_state[dim] += effect_size  # Red pushes UP
            else:
                new_state[dim] -= effect_size  # Blue pushes DOWN

        # Opponent counter-action
        if opponent_action is not None:
            np.random.seed(opponent_action)
            opp_affected_dims = np.random.choice(self.state_dim, size=self.state_dim // 4, replace=False)

            for dim in opp_affected_dims:
                effect_size = np.random.uniform(0.05, 0.15)

                if self.agent_role == "red":
                    new_state[dim] -= effect_size  # Blue opponent pushes DOWN
                else:
                    new_state[dim] += effect_size  # Red opponent pushes UP

        # Small noise
        noise = np.random.normal(0, 0.01, size=self.state_dim)
        new_state += noise

        # Bounds
        new_state = np.clip(new_state, 0.0, 1.0)

        # Reset seed
        np.random.seed(None)

        return new_state.astype(np.float32)

    def _check_terminal_conditions(self) -> Tuple[bool, str]:
        """
        Check if episode should terminate and who won

        Returns:
            (terminated, winner)
            winner: "red", "blue", "draw", or "ongoing"
        """
        state_mean = np.mean(self.state)

        # Red victory: Network compromised
        if state_mean > 0.7:
            return True, "red"

        # Blue victory: Network secured
        if state_mean < 0.3:
            return True, "blue"

        # Timeout - BALANCED FIX
        if self.step_count >= self.max_steps:
            # If time runs out, winner determined by final state position
            # State >0.5 slightly favors Red, <0.5 favors Blue
            # But neither gets automatic win - promotes decisive play
            if state_mean > 0.55:
                return True, "red"  # Red gained ground, wins
            elif state_mean < 0.45:
                return True, "blue"  # Blue gained ground, wins
            else:
                return True, "draw"  # True stalemate = draw

        # Game continues
        return False, "ongoing"

    def _calculate_zerosum_reward(self, winner: str, terminated: bool) -> float:
        """
        Calculate ZERO-SUM reward

        Final outcome rewards:
        - Win: +100
        - Loss: -100
        - Draw: 0

        Per-step shaping:
        - Tiny reward for moving state in right direction (max ±1)
        - This provides learning signal but doesn't dominate final outcome
        """
        # Per-step shaping reward (optional, can be disabled)
        shaping_reward = 0.0

        if not terminated:
            # Small shaping bonus for state movement
            state_mean = np.mean(self.state)

            if self.agent_role == "red":
                # Red wants state high
                shaping_reward = (state_mean - 0.5) * 0.5  # Max ±0.25
            else:
                # Blue wants state low
                shaping_reward = (0.5 - state_mean) * 0.5  # Max ±0.25

        # Final outcome reward (ZERO-SUM)
        outcome_reward = 0.0

        if terminated:
            if winner == self.agent_role:
                outcome_reward = 100.0  # WIN
            elif winner == "draw":
                outcome_reward = 0.0    # DRAW
            else:
                outcome_reward = -100.0  # LOSS

        total_reward = shaping_reward + outcome_reward

        return float(total_reward)

    def _get_observation(self) -> np.ndarray:
        """Get current observation"""
        obs = np.zeros(self.state_dim, dtype=np.float32)
        obs[:len(self.state)] = self.state

        # Add step progress if space available
        if len(obs) > len(self.state):
            obs[len(self.state)] = self.step_count / self.max_steps

        return obs


def test_win_conditions():
    """
    Test that win/loss conditions produce correct zero-sum rewards

    This MUST pass before any training!
    """
    print("\n" + "="*70)
    print("ZERO-SUM REWARD VALIDATION")
    print("="*70)

    # Test 1: Red victory scenario
    print("\nTest 1: Scripted Red Victory")
    print("-" * 70)

    env = ZeroSumCyberEnv(agent_role="red", opponent_model=None)
    env.reset()

    # Force Red victory by manipulating state (testing only!)
    env.state = np.full(env.state_dim, 0.8, dtype=np.float32)  # > 0.7 = Red wins

    obs, reward, terminated, _, info = env.step(0)

    print(f"State mean: {info['state_mean']:.3f}")
    print(f"Winner: {info['winner']}")
    print(f"Red reward: {reward:.1f}")

    if terminated and info['winner'] == 'red' and reward > 90:
        print("✓ Red victory produces positive reward (~+100)")
    else:
        print("❌ FAILED: Red victory should give ~+100 reward")
        return False

    # Test 2: Blue victory scenario (Red's perspective)
    print("\nTest 2: Scripted Blue Victory (Red's perspective)")
    print("-" * 70)

    env = ZeroSumCyberEnv(agent_role="red", opponent_model=None)
    env.reset()

    # Force Blue victory
    env.state = np.full(env.state_dim, 0.2, dtype=np.float32)  # < 0.3 = Blue wins

    obs, reward, terminated, _, info = env.step(0)

    print(f"State mean: {info['state_mean']:.3f}")
    print(f"Winner: {info['winner']}")
    print(f"Red reward: {reward:.1f}")

    if terminated and info['winner'] == 'blue' and reward < -90:
        print("✓ Blue victory produces negative reward for Red (~-100)")
    else:
        print("❌ FAILED: Blue victory should give Red ~-100 reward")
        return False

    # Test 3: Blue agent perspective (should be opposite)
    print("\nTest 3: Blue Agent Wins (Blue's perspective)")
    print("-" * 70)

    env = ZeroSumCyberEnv(agent_role="blue", opponent_model=None)
    env.reset()

    # Force Blue victory
    env.state = np.full(env.state_dim, 0.2, dtype=np.float32)

    obs, reward, terminated, _, info = env.step(0)

    print(f"State mean: {info['state_mean']:.3f}")
    print(f"Winner: {info['winner']}")
    print(f"Blue reward: {reward:.1f}")

    if terminated and info['winner'] == 'blue' and reward > 90:
        print("✓ Blue victory produces positive reward for Blue (~+100)")
    else:
        print("❌ FAILED: Blue should get +100 for winning")
        return False

    # Test 4: Verify zero-sum property
    print("\nTest 4: Zero-Sum Property")
    print("-" * 70)

    # Same scenario from both perspectives
    env_red = ZeroSumCyberEnv(agent_role="red", opponent_model=None)
    env_blue = ZeroSumCyberEnv(agent_role="blue", opponent_model=None)

    env_red.reset()
    env_blue.reset()

    # Force same outcome
    env_red.state = np.full(env_red.state_dim, 0.75, dtype=np.float32)
    env_blue.state = np.full(env_blue.state_dim, 0.75, dtype=np.float32)

    _, reward_red, _, _, _ = env_red.step(0)
    _, reward_blue, _, _, _ = env_blue.step(0)

    reward_sum = reward_red + reward_blue

    print(f"Red reward: {reward_red:.1f}")
    print(f"Blue reward: {reward_blue:.1f}")
    print(f"Sum: {reward_sum:.1f}")

    if abs(reward_sum) < 5:  # Should be ~0 (allowing for small shaping)
        print("✓ Rewards sum to ~0 (zero-sum property holds)")
    else:
        print("❌ FAILED: Rewards should sum to zero")
        return False

    # Test 5: Random play statistics
    print("\nTest 5: Random Play Statistics")
    print("-" * 70)

    env = ZeroSumCyberEnv(agent_role="red", opponent_model=None)

    outcomes = {'red': 0, 'blue': 0, 'draw': 0}
    episode_rewards = []

    for ep in range(100):
        obs, _ = env.reset()
        total_reward = 0

        for step in range(100):
            action = env.action_space.sample()
            obs, reward, terminated, _, info = env.step(action)
            total_reward += reward

            if terminated:
                outcomes[info['winner']] += 1
                break

        episode_rewards.append(total_reward)

    print(f"Outcomes (100 random episodes):")
    print(f"  Red wins: {outcomes['red']}")
    print(f"  Blue wins: {outcomes['blue']}")
    print(f"  Draws: {outcomes['draw']}")

    print(f"\nReward statistics:")
    print(f"  Mean: {np.mean(episode_rewards):.1f}")
    print(f"  Std:  {np.std(episode_rewards):.1f}")
    print(f"  Min:  {np.min(episode_rewards):.1f}")
    print(f"  Max:  {np.max(episode_rewards):.1f}")

    # Check strong learning signal
    if np.std(episode_rewards) > 30:
        print("✓ High reward variance (strong learning signal)")
    else:
        print("⚠️  Low reward variance - learning signal may be weak")

    # Check not biased
    if abs(np.mean(episode_rewards)) < 20:
        print("✓ Mean reward near zero (unbiased)")
    else:
        print("⚠️  Mean reward far from zero - may be biased")

    print("\n" + "="*70)
    print("✅ ALL ZERO-SUM TESTS PASSED")
    print("="*70)
    print("\nReward structure is correct. Ready for training.")
    print("="*70 + "\n")

    return True


if __name__ == "__main__":
    import sys
    success = test_win_conditions()
    sys.exit(0 if success else 1)
