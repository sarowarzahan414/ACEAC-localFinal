"""
Dynamic Co-Evolution with Real-World Threat Intelligence Integration

Combines pure learning philosophy with real-world threat data:
- Actions remain ABSTRACT (no hardcoded exploits)
- Reward shaping from MITRE ATT&CK patterns
- Evaluation against real APT campaigns
- Curriculum based on threat landscape
- Novel pattern discovery encouraged

Usage:
    python aceac_dynamic_with_threat_intel.py --generations 20

Author: @sarowarzahan414
Date: 2025-11-19
Philosophy: Learn from reality, discover beyond it
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import PPO
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple, Dict

# Import threat intelligence integration
from threat_intelligence_integration import ThreatIntelligenceIntegration


class DynamicCoEvolutionWithThreatIntel(gym.Env):
    """
    Dynamic co-evolution environment enhanced with threat intelligence.

    Key Differences from base system:
    1. Reward shaping: Small bonus for realistic TTP patterns
    2. Evaluation: Compare strategies to real-world threats
    3. Curriculum: Progressive complexity from threat landscape

    Key Similarities (Preserved Philosophy):
    1. Actions remain abstract (NO predetermined meanings)
    2. Agents explore freely (NO forced sequences)
    3. Novel patterns rewarded (NO over-fitting to known attacks)
    4. Pure learning (NO hardcoded effectiveness values)
    """

    metadata = {'render_modes': [], 'name': 'ACEAC-Dynamic-ThreatIntel-v1'}

    def __init__(self,
                 agent_role: str = "red",
                 opponent_model: Optional[object] = None,
                 num_actions: int = 25,
                 state_dim: int = 32,
                 enable_threat_intel: bool = True,
                 ttp_bonus_weight: float = 0.1):
        """
        Initialize environment with optional threat intelligence.

        Args:
            agent_role: "red" or "blue"
            opponent_model: Opponent agent (or None for solo training)
            num_actions: Number of available actions
            state_dim: Dimensionality of state space
            enable_threat_intel: Enable threat intel reward shaping
            ttp_bonus_weight: Weight for TTP alignment bonus (0.0-1.0)
        """
        super().__init__()

        self.agent_role = agent_role
        self.opponent_model = opponent_model
        self.num_actions = num_actions
        self.state_dim = state_dim
        self.enable_threat_intel = enable_threat_intel
        self.ttp_bonus_weight = ttp_bonus_weight

        # Observation and action spaces
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(state_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(num_actions)

        # State variables
        self.state = None
        self.step_count = 0
        self.max_steps = 100

        # Action history (for TTP analysis)
        self.action_history = []
        self.opponent_action_history = []

        # Threat intelligence (if enabled)
        if self.enable_threat_intel:
            self.threat_intel = ThreatIntelligenceIntegration(
                enable_mitre_attack=True,
                enable_nvd=False,  # Disable for speed (can enable with API key)
                enable_live_feeds=False
            )
        else:
            self.threat_intel = None

        self.reset()

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        """Reset environment to initial state"""
        super().reset(seed=seed)

        # Initialize state
        self.state = np.random.uniform(0.3, 0.7, size=self.state_dim)
        self.state = self.state.astype(np.float32)

        self.step_count = 0
        self.action_history = []
        self.opponent_action_history = []

        return self._get_observation(), {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute action with threat intel-enhanced rewards.

        The key difference: Adds small TTP alignment bonus to base reward.
        """
        self.step_count += 1

        # Validate action
        if not self.action_space.contains(action):
            action = self.action_space.sample()

        self.action_history.append(action)

        # Get opponent action
        opponent_action = None
        if self.opponent_model is not None:
            try:
                obs = self._get_observation()
                opponent_action, _ = self.opponent_model.predict(obs, deterministic=False)
                opponent_action = int(opponent_action)
                self.opponent_action_history.append(opponent_action)
            except:
                opponent_action = None

        # Apply state transition (same as base system - NO CHANGES)
        self.state = self._dynamic_state_transition(action, opponent_action)

        # Calculate BASE reward (same as base system - NO CHANGES)
        base_reward = self._calculate_base_reward(action, opponent_action)

        # ADD: Threat intel bonus (if enabled)
        ttp_bonus = 0.0
        if self.enable_threat_intel and len(self.action_history) >= 4:
            # Calculate TTP alignment bonus
            raw_ttp_bonus = self.threat_intel.calculate_ttp_alignment_bonus(
                action_sequence=self.action_history[-4:],
                max_bonus=10.0
            )

            # Weight the bonus (keep it small so it doesn't dominate)
            ttp_bonus = raw_ttp_bonus * self.ttp_bonus_weight

        # Total reward = base + weighted TTP bonus
        total_reward = base_reward + ttp_bonus

        # Termination
        terminated = self.step_count >= self.max_steps

        # Info
        info = {
            'step': self.step_count,
            'state_mean': float(np.mean(self.state)),
            'state_std': float(np.std(self.state)),
            'action': int(action),
            'opponent_action': int(opponent_action) if opponent_action is not None else -1,
            'base_reward': float(base_reward),
            'ttp_bonus': float(ttp_bonus),
            'total_reward': float(total_reward)
        }

        return self._get_observation(), float(total_reward), terminated, False, info

    def _dynamic_state_transition(self, action: int, opponent_action: Optional[int]) -> np.ndarray:
        """
        State transition - IDENTICAL to base system.
        NO CHANGES for threat intel integration.
        """
        new_state = self.state.copy()

        # Action influence
        np.random.seed(action)
        affected_dims = np.random.choice(self.state_dim, size=self.state_dim // 4, replace=False)

        for dim in affected_dims:
            effect_size = np.random.uniform(0.05, 0.15)
            direction = 1.0 if self.agent_role == "red" else -1.0
            new_state[dim] += direction * effect_size

        # Opponent influence
        if opponent_action is not None:
            np.random.seed(opponent_action)
            opp_affected_dims = np.random.choice(self.state_dim, size=self.state_dim // 4, replace=False)

            for dim in opp_affected_dims:
                effect_size = np.random.uniform(0.05, 0.15)
                direction = -1.0 if self.agent_role == "red" else 1.0
                new_state[dim] += direction * effect_size

        # Natural dynamics
        noise = np.random.normal(0, 0.02, size=self.state_dim)
        new_state += noise

        # Bounds
        new_state = np.clip(new_state, 0.0, 1.0)

        # Reset seed
        np.random.seed(None)

        return new_state.astype(np.float32)

    def _calculate_base_reward(self, action: int, opponent_action: Optional[int]) -> float:
        """
        Base reward calculation - IDENTICAL to base system.
        NO CHANGES for threat intel integration.
        """
        # State reward
        if len(self.action_history) > 1:
            prev_state_mean = 0.5
            if hasattr(self, '_prev_state_mean'):
                prev_state_mean = self._prev_state_mean

            current_state_mean = np.mean(self.state)

            if self.agent_role == "red":
                state_reward = (current_state_mean - prev_state_mean) * 100.0
            else:
                state_reward = (prev_state_mean - current_state_mean) * 100.0

            self._prev_state_mean = current_state_mean
        else:
            state_reward = 0.0
            self._prev_state_mean = np.mean(self.state)

        # Diversity bonus
        diversity_reward = self._diversity_bonus()

        # Interaction reward
        interaction_reward = 0.0
        if opponent_action is not None:
            state_advantage = np.mean(self.state) - 0.5
            if self.agent_role == "red":
                interaction_reward = state_advantage * 10.0
            else:
                interaction_reward = -state_advantage * 10.0

        # Total base reward
        total_reward = state_reward + diversity_reward + interaction_reward

        # Clip
        total_reward = np.clip(total_reward, -50.0, 50.0)

        return float(total_reward)

    def _diversity_bonus(self) -> float:
        """Diversity bonus - IDENTICAL to base system"""
        if len(self.action_history) < 10:
            return 0.0

        recent_actions = self.action_history[-10:]
        unique_actions = len(set(recent_actions))
        diversity_ratio = unique_actions / 10.0
        bonus = diversity_ratio * 5.0

        return bonus

    def _get_observation(self) -> np.ndarray:
        """Get observation - IDENTICAL to base system"""
        obs = np.zeros(self.state_dim, dtype=np.float32)
        obs[:len(self.state)] = self.state

        if len(obs) > len(self.state):
            obs[len(self.state)] = self.step_count / self.max_steps

        if len(obs) > len(self.state) + 1 and len(self.action_history) > 0:
            last_action_normalized = self.action_history[-1] / self.num_actions
            if len(self.state) + 1 < len(obs):
                obs[len(self.state) + 1] = last_action_normalized

        return obs

    def get_threat_intel_evaluation(self) -> Dict:
        """
        Evaluate strategy realism (post-episode analysis).

        This is called AFTER training for reporting, NOT during.
        """
        if not self.enable_threat_intel:
            return {'enabled': False}

        return self.threat_intel.evaluate_strategy_realism(
            action_history=self.action_history,
            opponent_action_history=self.opponent_action_history
        )


class DynamicCoEvolutionTrainerWithThreatIntel:
    """
    Trainer with threat intelligence integration.

    Adds:
    1. Curriculum learning (progressive TTP bonus weight)
    2. Threat reports after each generation
    3. Realism evaluation
    """

    def __init__(self,
                 num_actions: int = 25,
                 state_dim: int = 32,
                 population_size: int = 5,
                 enable_threat_intel: bool = True):
        """Initialize trainer with threat intel"""
        self.num_actions = num_actions
        self.state_dim = state_dim
        self.population_size = population_size
        self.enable_threat_intel = enable_threat_intel

        self.red_population = []
        self.blue_population = []
        self.generation_history = []

        # Threat intelligence
        if self.enable_threat_intel:
            self.threat_intel = ThreatIntelligenceIntegration(
                enable_mitre_attack=True,
                enable_nvd=False,
                enable_live_feeds=False
            )

    def train(self,
              num_generations: int = 20,
              steps_per_generation: int = 50000,
              save_dir: str = "models/dynamic_threat_intel"):
        """Train with threat intelligence integration"""

        print("\n" + "="*70)
        print("DYNAMIC CO-EVOLUTION WITH THREAT INTELLIGENCE")
        print("="*70)
        print(f"Generations: {num_generations}")
        print(f"Steps per generation: {steps_per_generation}")
        print(f"Threat Intel: {'ENABLED' if self.enable_threat_intel else 'DISABLED'}")
        print("="*70 + "\n")

        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        # Initialize agents
        print("Initializing agents...")

        red_env = DynamicCoEvolutionWithThreatIntel(
            agent_role="red",
            opponent_model=None,
            num_actions=self.num_actions,
            state_dim=self.state_dim,
            enable_threat_intel=self.enable_threat_intel,
            ttp_bonus_weight=0.1  # Start with small weight
        )

        blue_env = DynamicCoEvolutionWithThreatIntel(
            agent_role="blue",
            opponent_model=None,
            num_actions=self.num_actions,
            state_dim=self.state_dim,
            enable_threat_intel=self.enable_threat_intel,
            ttp_bonus_weight=0.1
        )

        red_model = PPO("MlpPolicy", red_env, verbose=0)
        blue_model = PPO("MlpPolicy", blue_env, verbose=0)

        print("✓ Agents initialized\n")

        # Training loop
        for gen in range(1, num_generations + 1):
            print(f"{'='*70}")
            print(f"GENERATION {gen}/{num_generations}")
            print(f"{'='*70}\n")

            # Get curriculum complexity
            if self.enable_threat_intel:
                curriculum = self.threat_intel.get_curriculum_complexity(gen)
                ttp_weight = curriculum['suggested_bonus_weight']
                print(f"Curriculum: {curriculum['complexity']}")
                print(f"TTP Bonus Weight: {ttp_weight:.3f}")
                print(f"Focus Tactics: {', '.join(curriculum['focus_tactics'][:3])}...\n")
            else:
                ttp_weight = 0.0

            # Update environment TTP weights
            blue_opponent = self._select_opponent(self.blue_population)
            red_opponent = self._select_opponent(self.red_population)

            # Train Red
            print(f"Training Red agent...")
            red_env = DynamicCoEvolutionWithThreatIntel(
                agent_role="red",
                opponent_model=blue_opponent,
                num_actions=self.num_actions,
                state_dim=self.state_dim,
                enable_threat_intel=self.enable_threat_intel,
                ttp_bonus_weight=ttp_weight
            )
            red_model.set_env(red_env)
            red_model.learn(total_timesteps=steps_per_generation, progress_bar=True)
            print("✓ Red training complete\n")

            # Train Blue
            print(f"Training Blue agent...")
            blue_env = DynamicCoEvolutionWithThreatIntel(
                agent_role="blue",
                opponent_model=red_model,
                num_actions=self.num_actions,
                state_dim=self.state_dim,
                enable_threat_intel=self.enable_threat_intel,
                ttp_bonus_weight=ttp_weight
            )
            blue_model.set_env(blue_env)
            blue_model.learn(total_timesteps=steps_per_generation, progress_bar=True)
            print("✓ Blue training complete\n")

            # Evaluate
            print(f"Evaluating generation {gen}...")
            eval_results = self._evaluate_generation(red_model, blue_model)

            # Threat intelligence report
            if self.enable_threat_intel:
                report = self.threat_intel.generate_threat_report(
                    red_history=eval_results.get('red_actions', []),
                    blue_history=eval_results.get('blue_actions', []),
                    generation=gen
                )
                print(report)

            # Update populations
            self.red_population.append(red_model)
            self.blue_population.append(blue_model)

            if len(self.red_population) > self.population_size:
                self.red_population.pop(0)
            if len(self.blue_population) > self.population_size:
                self.blue_population.pop(0)

            # Save checkpoints
            red_model.save(save_path / f"red_gen_{gen}.zip")
            blue_model.save(save_path / f"blue_gen_{gen}.zip")

            # Record generation
            gen_data = {
                'generation': gen,
                'evaluation': eval_results,
                'curriculum': curriculum if self.enable_threat_intel else {},
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
                'threat_intel_enabled': self.enable_threat_intel,
                'generations': self.generation_history
            }, f, indent=2)

        print(f"✓ Training history saved: {history_file}\n")

        return red_model, blue_model

    def _select_opponent(self, population):
        """Select opponent from population"""
        if not population:
            return None
        return np.random.choice(population)

    def _evaluate_generation(self, red_model, blue_model, num_episodes: int = 10):
        """Evaluate generation"""
        state_values = []
        red_actions = []
        blue_actions = []

        for ep in range(num_episodes):
            env = DynamicCoEvolutionWithThreatIntel(
                agent_role="red",
                opponent_model=blue_model,
                num_actions=self.num_actions,
                state_dim=self.state_dim,
                enable_threat_intel=self.enable_threat_intel
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

        results = {
            'state_dominance': float(np.mean(state_values)),
            'red_diversity': len(set(red_actions)) / len(red_actions) if red_actions else 0.0,
            'blue_diversity': len(set(blue_actions)) / len(blue_actions) if blue_actions else 0.0,
            'red_actions': red_actions,
            'blue_actions': blue_actions
        }

        return results


def main():
    """Main training function"""
    import argparse

    parser = argparse.ArgumentParser(description='Dynamic Co-Evolution with Threat Intel')
    parser.add_argument('--generations', type=int, default=20, help='Number of generations')
    parser.add_argument('--steps', type=int, default=50000, help='Steps per generation')
    parser.add_argument('--no-threat-intel', action='store_true', help='Disable threat intel')

    args = parser.parse_args()

    trainer = DynamicCoEvolutionTrainerWithThreatIntel(
        num_actions=25,
        state_dim=32,
        population_size=5,
        enable_threat_intel=not args.no_threat_intel
    )

    red_model, blue_model = trainer.train(
        num_generations=args.generations,
        steps_per_generation=args.steps,
        save_dir="models/dynamic_threat_intel"
    )

    print("="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print("\nAgents learned through:")
    print("  ✓ Pure exploration (no hardcoded attacks)")
    print("  ✓ Reward shaping from real threat intelligence")
    print("  ✓ Self-play co-evolution")
    print("\nResult: Realistic attack patterns + novel discoveries")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
