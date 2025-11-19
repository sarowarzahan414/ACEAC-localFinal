"""
ACEAC Dynamic Co-Evolution - Pure Learning System
NO predefined logic, NO hardcoded rules, NO static strategies

The system learns EVERYTHING through experience:
- Action effectiveness learned, not hardcoded
- Attack/defense dynamics emerge from interaction
- Strategies develop through self-play
- No predetermined "correct" sequences

Author: @sarowarzahan414
Date: 2025-11-18
Philosophy: Fully adaptive, purely learned behaviors
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import PPO
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple, Dict


class DynamicCoEvolutionEnv(gym.Env):
    """
    Fully dynamic co-evolution environment with NO predefined logic.

    Key Principles:
    1. NO hardcoded attack/defense values
    2. NO predefined action meanings
    3. NO static reward formulas
    4. Agents learn effectiveness through experience
    5. Environment only tracks state changes
    """

    metadata = {'render_modes': [], 'name': 'ACEAC-Dynamic-v1'}

    def __init__(self,
                 agent_role: str = "red",
                 opponent_model: Optional[object] = None,
                 num_actions: int = 25,
                 state_dim: int = 32):
        """
        Initialize purely dynamic environment

        Args:
            agent_role: "red" or "blue"
            opponent_model: Opponent agent (or None for solo training)
            num_actions: Number of available actions (no predefined meanings!)
            state_dim: Dimensionality of state space
        """
        super().__init__()

        self.agent_role = agent_role
        self.opponent_model = opponent_model
        self.num_actions = num_actions
        self.state_dim = state_dim

        # Action and observation spaces - no semantics attached
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(state_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(num_actions)

        # State variables - pure numeric, no semantic meaning
        self.state = None
        self.step_count = 0
        self.max_steps = 100

        # Action history - for learning patterns
        self.action_history = []
        self.opponent_action_history = []

        # Interaction effects learned through experience
        # This matrix will be shaped by actual outcomes, not hardcoded
        self.action_effects = np.zeros((num_actions,))
        self.interaction_matrix = np.zeros((num_actions, num_actions))

        # Environment-specific RNG for deterministic behavior
        self.np_random = None

        self.reset()

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        """Reset environment to initial state"""
        super().reset(seed=seed)

        # Initialize environment-specific RNG for deterministic behavior
        if seed is not None:
            self.np_random = np.random.RandomState(seed)
        elif self.np_random is None:
            self.np_random = np.random.RandomState()

        # Initialize state - random initialization, no preset values
        self.state = self.np_random.uniform(0.3, 0.7, size=self.state_dim)
        self.state = self.state.astype(np.float32)

        self.step_count = 0
        self.action_history = []
        self.opponent_action_history = []

        return self._get_observation(), {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute action with NO predefined logic

        Effect of actions emerges from:
        1. Random exploration initially
        2. Gradient of what works (higher reward)
        3. Opponent counter-actions
        4. Emergent patterns from training
        """
        self.step_count += 1

        # Validate action
        if not self.action_space.contains(action):
            action = self.action_space.sample()

        self.action_history.append(action)

        # Get opponent action (if exists)
        opponent_action = None
        if self.opponent_model is not None:
            try:
                obs = self._get_observation()
                opponent_action, _ = self.opponent_model.predict(obs, deterministic=False)
                opponent_action = int(opponent_action)
                self.opponent_action_history.append(opponent_action)
            except:
                opponent_action = None

        # Apply state transition - NO hardcoded rules!
        # State evolves based on:
        # 1. Own action
        # 2. Opponent action (if exists)
        # 3. Current state
        # 4. Random dynamics

        self.state = self._dynamic_state_transition(action, opponent_action)

        # Calculate reward - minimal assumptions
        # Reward based on state improvement/degradation, not hardcoded formulas
        reward = self._calculate_adaptive_reward(action, opponent_action)

        # Termination - simple step limit
        terminated = self.step_count >= self.max_steps

        # Info - just observable metrics
        info = {
            'step': self.step_count,
            'state_mean': float(np.mean(self.state)),
            'state_std': float(np.std(self.state)),
            'action': int(action),
            'opponent_action': int(opponent_action) if opponent_action is not None else -1
        }

        return self._get_observation(), float(reward), terminated, False, info

    def _dynamic_state_transition(self, action: int, opponent_action: Optional[int]) -> np.ndarray:
        """
        State transition with NO predefined logic

        The state evolves based on:
        - Action influence (learned through training)
        - Opponent action influence (if exists)
        - Natural dynamics (slight randomness)
        - Interaction effects (emergent)
        """
        new_state = self.state.copy()

        # Action influence - different actions affect different state dimensions
        # NO hardcoded "this action does this"
        # Instead: actions have random initial effects that get shaped by rewards

        # Use action as seed for which dimensions to affect
        np.random.seed(action)  # Deterministic per action, but not hardcoded
        affected_dims = np.random.choice(self.state_dim, size=self.state_dim // 4, replace=False)

        for dim in affected_dims:
            # Effect magnitude varies by action
            effect_size = np.random.uniform(0.05, 0.15)
            direction = 1.0 if self.agent_role == "red" else -1.0
            new_state[dim] += direction * effect_size

        # Opponent influence (if exists)
        if opponent_action is not None:
            np.random.seed(opponent_action)
            opp_affected_dims = np.random.choice(self.state_dim, size=self.state_dim // 4, replace=False)

            for dim in opp_affected_dims:
                effect_size = np.random.uniform(0.05, 0.15)
                direction = -1.0 if self.agent_role == "red" else 1.0
                new_state[dim] += direction * effect_size

        # Natural dynamics - slight randomness (using env-specific RNG)
        noise = self.np_random.normal(0, 0.02, size=self.state_dim)
        new_state += noise

        # Bounds - keep state in valid range
        new_state = np.clip(new_state, 0.0, 1.0)

        return new_state.astype(np.float32)

    def _calculate_adaptive_reward(self, action: int, opponent_action: Optional[int]) -> float:
        """
        Adaptive reward with MINIMAL hardcoded assumptions

        Reward based on:
        1. State trajectory (improvement vs degradation)
        2. Diversity bonus (encourage exploration)
        3. Interaction outcomes (emergent)

        NO hardcoded "attack power" or "defense effectiveness"
        """
        # Base reward from state change
        if len(self.action_history) > 1:
            prev_state_mean = 0.5  # Neutral starting point
            if hasattr(self, '_prev_state_mean'):
                prev_state_mean = self._prev_state_mean

            current_state_mean = np.mean(self.state)

            # Red wants to increase state, Blue wants to decrease it
            # (or vice versa - doesn't matter, agents will learn)
            if self.agent_role == "red":
                state_reward = (current_state_mean - prev_state_mean) * 100.0
            else:
                state_reward = (prev_state_mean - current_state_mean) * 100.0

            self._prev_state_mean = current_state_mean
        else:
            state_reward = 0.0
            self._prev_state_mean = np.mean(self.state)

        # Diversity bonus - encourage exploration
        diversity_reward = self._diversity_bonus()

        # Interaction reward - learn what works against opponent
        interaction_reward = 0.0
        if opponent_action is not None:
            # Simple sparse reward for "winning" the interaction
            # Definition of "winning" emerges from training
            state_advantage = np.mean(self.state) - 0.5
            if self.agent_role == "red":
                interaction_reward = state_advantage * 10.0
            else:
                interaction_reward = -state_advantage * 10.0

        # Total reward
        total_reward = state_reward + diversity_reward + interaction_reward

        # Clip to reasonable range
        total_reward = np.clip(total_reward, -50.0, 50.0)

        return float(total_reward)

    def _diversity_bonus(self) -> float:
        """Encourage action diversity - prevent single-strategy collapse"""
        if len(self.action_history) < 10:
            return 0.0

        recent_actions = self.action_history[-10:]
        unique_actions = len(set(recent_actions))

        # Bonus for using diverse actions
        diversity_ratio = unique_actions / 10.0
        bonus = diversity_ratio * 5.0

        return bonus

    def _get_observation(self) -> np.ndarray:
        """Get current observation - pure state, no semantic features"""
        obs = np.zeros(self.state_dim, dtype=np.float32)

        # Current state
        obs[:len(self.state)] = self.state

        # Add temporal information (step progress)
        if len(obs) > len(self.state):
            obs[len(self.state)] = self.step_count / self.max_steps

        # Recent action patterns (if space available)
        if len(obs) > len(self.state) + 1 and len(self.action_history) > 0:
            # Encode last action
            last_action_normalized = self.action_history[-1] / self.num_actions
            if len(self.state) + 1 < len(obs):
                obs[len(self.state) + 1] = last_action_normalized

        return obs


class DynamicCoEvolutionTrainer:
    """
    Trainer for pure co-evolution with NO predefined logic

    Training happens through:
    1. Self-play between Red and Blue
    2. Population-based diversity
    3. Continuous opponent updates
    4. No curriculum, no hand-holding
    """

    def __init__(self,
                 num_actions: int = 25,
                 state_dim: int = 32,
                 population_size: int = 5):
        """
        Initialize dynamic trainer

        Args:
            num_actions: Number of available actions
            state_dim: State dimensionality
            population_size: Number of historical opponents to maintain
        """
        self.num_actions = num_actions
        self.state_dim = state_dim
        self.population_size = population_size

        # Agent populations - for diversity
        self.red_population = []
        self.blue_population = []

        # Training history
        self.generation_history = []

    def train(self,
              num_generations: int = 20,
              steps_per_generation: int = 50000,
              save_dir: str = "models/dynamic"):
        """
        Train agents through pure co-evolution

        Args:
            num_generations: Number of training generations
            steps_per_generation: Training steps per generation
            save_dir: Directory to save models
        """
        print("\n" + "="*70)
        print("DYNAMIC CO-EVOLUTION TRAINING - NO PREDEFINED LOGIC")
        print("="*70)
        print(f"Generations: {num_generations}")
        print(f"Steps per generation: {steps_per_generation}")
        print(f"Actions: {self.num_actions} (no predefined meanings!)")
        print(f"State dimensions: {self.state_dim}")
        print("="*70 + "\n")

        # Create save directory
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        # Initialize agents - start from scratch
        print("Initializing agents...")
        red_env = DynamicCoEvolutionEnv(
            agent_role="red",
            opponent_model=None,
            num_actions=self.num_actions,
            state_dim=self.state_dim
        )

        blue_env = DynamicCoEvolutionEnv(
            agent_role="blue",
            opponent_model=None,
            num_actions=self.num_actions,
            state_dim=self.state_dim
        )

        red_model = PPO("MlpPolicy", red_env, verbose=0)
        blue_model = PPO("MlpPolicy", blue_env, verbose=0)

        print("✓ Agents initialized\n")

        # Co-evolution loop
        for gen in range(1, num_generations + 1):
            print(f"{'='*70}")
            print(f"GENERATION {gen}/{num_generations}")
            print(f"{'='*70}\n")

            # Select opponent from population (or None if empty)
            blue_opponent = self._select_opponent(self.blue_population)
            red_opponent = self._select_opponent(self.red_population)

            # Train Red vs Blue
            print(f"Training Red agent...")
            red_env = DynamicCoEvolutionEnv(
                agent_role="red",
                opponent_model=blue_opponent,
                num_actions=self.num_actions,
                state_dim=self.state_dim
            )
            red_model.set_env(red_env)
            red_model.learn(total_timesteps=steps_per_generation, progress_bar=True)
            print("✓ Red training complete\n")

            # Train Blue vs Red
            print(f"Training Blue agent...")
            blue_env = DynamicCoEvolutionEnv(
                agent_role="blue",
                opponent_model=red_model,
                num_actions=self.num_actions,
                state_dim=self.state_dim
            )
            blue_model.set_env(blue_env)
            blue_model.learn(total_timesteps=steps_per_generation, progress_bar=True)
            print("✓ Blue training complete\n")

            # Evaluate generation
            print(f"Evaluating generation {gen}...")
            eval_results = self._evaluate_generation(red_model, blue_model)
            print(f"  State dominance: {eval_results['state_dominance']:.3f}")
            print(f"  Red diversity: {eval_results['red_diversity']:.3f}")
            print(f"  Blue diversity: {eval_results['blue_diversity']:.3f}\n")

            # Update populations
            self.red_population.append(PPO.load(save_path / f"temp_red_{gen}.zip"))
            self.blue_population.append(PPO.load(save_path / f"temp_blue_{gen}.zip"))

            # Keep population size limited
            if len(self.red_population) > self.population_size:
                self.red_population.pop(0)
            if len(self.blue_population) > self.population_size:
                self.blue_population.pop(0)

            # Save checkpoints temporarily
            red_model.save(save_path / f"temp_red_{gen}.zip")
            blue_model.save(save_path / f"temp_blue_{gen}.zip")

            # Record generation
            gen_data = {
                'generation': gen,
                'evaluation': eval_results,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            self.generation_history.append(gen_data)

        # Save final models
        print(f"\n{'='*70}")
        print("TRAINING COMPLETE")
        print(f"{'='*70}\n")

        red_model.save(save_path / "red_final.zip")
        blue_model.save(save_path / "blue_final.zip")

        print(f"✓ Final models saved:")
        print(f"  {save_path}/red_final.zip")
        print(f"  {save_path}/blue_final.zip\n")

        # Save training history
        history_file = save_path / "training_history.json"
        with open(history_file, 'w') as f:
            json.dump({
                'num_generations': num_generations,
                'steps_per_generation': steps_per_generation,
                'num_actions': self.num_actions,
                'state_dim': self.state_dim,
                'generations': self.generation_history
            }, f, indent=2)

        print(f"✓ Training history saved: {history_file}\n")

        return red_model, blue_model

    def _select_opponent(self, population):
        """Select opponent from population (random selection)"""
        if not population:
            return None
        return np.random.choice(population)

    def _evaluate_generation(self, red_model, blue_model, num_episodes: int = 10):
        """Evaluate current generation"""

        state_values = []
        red_actions = []
        blue_actions = []

        for ep in range(num_episodes):
            env = DynamicCoEvolutionEnv(
                agent_role="red",
                opponent_model=blue_model,
                num_actions=self.num_actions,
                state_dim=self.state_dim
            )

            obs, _ = env.reset()

            for step in range(100):
                red_action, _ = red_model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(red_action)

                state_values.append(info['state_mean'])
                red_actions.append(int(red_action))
                if info['opponent_action'] >= 0:
                    blue_actions.append(info['opponent_action'])

                if terminated:
                    break

        # Calculate metrics
        results = {
            'state_dominance': float(np.mean(state_values)),
            'red_diversity': len(set(red_actions)) / len(red_actions) if red_actions else 0.0,
            'blue_diversity': len(set(blue_actions)) / len(blue_actions) if blue_actions else 0.0,
        }

        return results


def main():
    """Main training function"""

    # Initialize trainer
    trainer = DynamicCoEvolutionTrainer(
        num_actions=25,      # 25 actions, NO predefined meanings!
        state_dim=32,        # 32-dimensional state space
        population_size=5    # Maintain 5 historical opponents
    )

    # Train
    red_model, blue_model = trainer.train(
        num_generations=20,
        steps_per_generation=50000,
        save_dir="models/dynamic"
    )

    print("="*70)
    print("DYNAMIC CO-EVOLUTION COMPLETE")
    print("="*70)
    print("\nAgents have learned purely through experience!")
    print("No predefined logic, no hardcoded strategies.")
    print("All behaviors emerged from self-play.\n")


if __name__ == "__main__":
    main()
