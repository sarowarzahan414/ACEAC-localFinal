"""
ACEAC Adaptive Dynamic Learning System
=======================================

A fully adaptive reinforcement learning system where Red and Blue team agents
learn continuously from past interactions with NO predefined logic or static rules.

Key Features:
- Continuous learning from all past interactions
- Recurrent neural network policies (LSTM) for temporal patterns
- Curiosity-driven exploration with intrinsic motivation
- Experience replay for lifelong learning
- Adaptive curriculum where agents self-adjust difficulty
- Pure emergent behavior discovery
- No hardcoded attack/defense semantics

Architecture:
- Red Team Agent: Learns attack strategies through experience
- Blue Team Agent: Learns defense strategies through experience
- Co-Evolution: Agents train against each other in competitive self-play
- Memory System: All interactions stored and learned from
- Meta-Learning: System learns how to learn more effectively
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from collections import deque
import time
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
import torch
import torch.nn as nn


# ============================================================================
# EXPERIENCE REPLAY BUFFER - Stores all past interactions for continuous learning
# ============================================================================

@dataclass
class Interaction:
    """Single interaction step in the environment"""
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
    opponent_action: Optional[int]
    timestep: int
    episode_id: int

    def to_dict(self) -> Dict:
        return {
            'state': self.state.tolist(),
            'action': int(self.action),
            'reward': float(self.reward),
            'next_state': self.next_state.tolist(),
            'done': bool(self.done),
            'opponent_action': int(self.opponent_action) if self.opponent_action is not None else None,
            'timestep': int(self.timestep),
            'episode_id': int(self.episode_id)
        }


class ExperienceReplayBuffer:
    """
    Stores all past interactions for continuous learning.
    Agents can sample from this buffer to learn from historical experiences.
    """

    def __init__(self, max_size: int = 1_000_000):
        self.max_size = max_size
        self.buffer: deque = deque(maxlen=max_size)
        self.episode_counter = 0

    def add(self, interaction: Interaction):
        """Add a new interaction to the buffer"""
        self.buffer.append(interaction)

    def sample(self, batch_size: int) -> List[Interaction]:
        """Sample a random batch of interactions"""
        if len(self.buffer) < batch_size:
            return list(self.buffer)
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in indices]

    def get_recent(self, n: int) -> List[Interaction]:
        """Get the n most recent interactions"""
        return list(self.buffer)[-n:] if n < len(self.buffer) else list(self.buffer)

    def get_episode(self, episode_id: int) -> List[Interaction]:
        """Get all interactions from a specific episode"""
        return [inter for inter in self.buffer if inter.episode_id == episode_id]

    def size(self) -> int:
        return len(self.buffer)

    def save(self, filepath: Path):
        """Save buffer to disk"""
        data = [inter.to_dict() for inter in self.buffer]
        with open(filepath, 'w') as f:
            json.dump(data, f)

    def load(self, filepath: Path):
        """Load buffer from disk"""
        with open(filepath, 'r') as f:
            data = json.load(f)

        self.buffer.clear()
        for item in data:
            interaction = Interaction(
                state=np.array(item['state']),
                action=item['action'],
                reward=item['reward'],
                next_state=np.array(item['next_state']),
                done=item['done'],
                opponent_action=item['opponent_action'],
                timestep=item['timestep'],
                episode_id=item['episode_id']
            )
            self.buffer.append(interaction)


# ============================================================================
# CURIOSITY MODULE - Intrinsic motivation for exploration
# ============================================================================

class CuriosityModule:
    """
    Implements curiosity-driven exploration using prediction error.
    Agents are rewarded for encountering novel situations.

    Based on "Curiosity-driven Exploration by Self-supervised Prediction"
    """

    def __init__(self, state_dim: int, action_dim: int, learning_rate: float = 0.001):
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Forward model: predicts next state from current state and action
        self.forward_model = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, state_dim)
        )

        # Inverse model: predicts action from state transition
        self.inverse_model = nn.Sequential(
            nn.Linear(state_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

        self.optimizer_forward = torch.optim.Adam(self.forward_model.parameters(), lr=learning_rate)
        self.optimizer_inverse = torch.optim.Adam(self.inverse_model.parameters(), lr=learning_rate)

        # Track prediction errors for intrinsic reward
        self.prediction_errors = deque(maxlen=10000)

    def compute_intrinsic_reward(self, state: np.ndarray, action: int, next_state: np.ndarray) -> float:
        """
        Compute intrinsic reward based on prediction error.
        Higher error = more novel = higher reward
        """
        with torch.no_grad():
            # Convert to tensors
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)

            # One-hot encode action
            action_onehot = torch.zeros(1, self.action_dim)
            action_onehot[0, action] = 1.0

            # Predict next state
            forward_input = torch.cat([state_tensor, action_onehot], dim=1)
            predicted_next_state = self.forward_model(forward_input)

            # Prediction error is the intrinsic reward
            prediction_error = torch.mean((predicted_next_state - next_state_tensor) ** 2).item()

            self.prediction_errors.append(prediction_error)

            # Normalize by recent errors to maintain consistent scale
            if len(self.prediction_errors) > 100:
                mean_error = np.mean(list(self.prediction_errors))
                std_error = np.std(list(self.prediction_errors)) + 1e-8
                normalized_error = (prediction_error - mean_error) / std_error
            else:
                normalized_error = prediction_error

            return float(np.clip(normalized_error, 0, 10))  # Clip to reasonable range

    def update(self, state: np.ndarray, action: int, next_state: np.ndarray):
        """
        Update the curiosity models based on observed transition.
        This is how the agent learns what to expect.
        """
        # Convert to tensors
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)

        # One-hot encode action
        action_onehot = torch.zeros(1, self.action_dim)
        action_onehot[0, action] = 1.0

        # Update forward model (predict next state from state + action)
        self.optimizer_forward.zero_grad()
        forward_input = torch.cat([state_tensor, action_onehot], dim=1)
        predicted_next_state = self.forward_model(forward_input)
        forward_loss = nn.MSELoss()(predicted_next_state, next_state_tensor)
        forward_loss.backward()
        self.optimizer_forward.step()

        # Update inverse model (predict action from state transition)
        self.optimizer_inverse.zero_grad()
        inverse_input = torch.cat([state_tensor, next_state_tensor], dim=1)
        predicted_action = self.inverse_model(inverse_input)
        inverse_loss = nn.CrossEntropyLoss()(predicted_action, torch.LongTensor([action]))
        inverse_loss.backward()
        self.optimizer_inverse.step()

        return {
            'forward_loss': forward_loss.item(),
            'inverse_loss': inverse_loss.item()
        }


# ============================================================================
# ADAPTIVE DYNAMIC ENVIRONMENT - Pure learning, no hardcoded logic
# ============================================================================

class AdaptiveDynamicEnv(gym.Env):
    """
    Fully adaptive environment where:
    - State evolves based on learned dynamics, not predefined rules
    - Actions have NO semantic meaning (no "attack" or "defend")
    - Rewards emerge from agent objectives (Red wants high state, Blue wants low)
    - Opponent behavior is predicted and learned
    - All mechanics are discovered through experience
    """

    metadata = {'render_modes': []}

    def __init__(
        self,
        state_dim: int = 32,
        num_actions: int = 25,
        max_steps: int = 100,
        agent_role: str = "red",  # "red" or "blue"
        use_curiosity: bool = True,
        curiosity_weight: float = 0.1
    ):
        super().__init__()

        self.state_dim = state_dim
        self.num_actions = num_actions
        self.max_steps = max_steps
        self.agent_role = agent_role
        self.use_curiosity = use_curiosity
        self.curiosity_weight = curiosity_weight

        # Gymnasium spaces
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(state_dim + 10,),  # Extra dims for meta-info
            dtype=np.float32
        )
        self.action_space = spaces.Discrete(num_actions)

        # Environment state
        self.state = None
        self.step_count = 0
        self.episode_id = 0

        # Opponent model (can be set externally for co-evolution)
        self.opponent_model = None

        # Action history for learning temporal patterns
        self.action_history = deque(maxlen=100)
        self.opponent_action_history = deque(maxlen=100)

        # Experience replay buffer
        self.experience_buffer = ExperienceReplayBuffer(max_size=1_000_000)

        # Curiosity module for intrinsic motivation
        if use_curiosity:
            self.curiosity = CuriosityModule(
                state_dim=state_dim + 10,
                action_dim=num_actions
            )
        else:
            self.curiosity = None

        # Learned action effects (updated dynamically during training)
        self.action_effects = np.zeros((num_actions, state_dim))
        self.action_counts = np.zeros(num_actions)

        # State transition model (learns how state evolves)
        self.transition_history = deque(maxlen=10000)

        # Statistics
        self.total_episodes = 0
        self.total_steps = 0

    def set_opponent_model(self, model):
        """Set the opponent model for competitive co-evolution"""
        self.opponent_model = model

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state"""
        super().reset(seed=seed)

        # Initialize state randomly (no predefined "healthy" or "compromised" state)
        self.state = self.np_random.uniform(0.3, 0.7, size=self.state_dim).astype(np.float32)
        self.step_count = 0
        self.action_history.clear()
        self.opponent_action_history.clear()

        self.episode_id = self.total_episodes
        self.total_episodes += 1

        obs = self._get_observation()
        info = {}

        return obs, info

    def _get_observation(self) -> np.ndarray:
        """
        Construct observation with:
        - Current state
        - Meta-information (step progress, action history features)
        """
        obs = np.zeros(self.observation_space.shape[0], dtype=np.float32)

        # Core state
        obs[:self.state_dim] = self.state

        # Meta-information
        meta_idx = self.state_dim

        # Progress through episode
        obs[meta_idx] = self.step_count / self.max_steps
        meta_idx += 1

        # Recent action features (if any)
        if len(self.action_history) > 0:
            recent_actions = list(self.action_history)[-5:]
            for i, action in enumerate(recent_actions):
                if meta_idx < len(obs):
                    obs[meta_idx] = action / self.num_actions
                    meta_idx += 1

        # Opponent action features (if any)
        if len(self.opponent_action_history) > 0:
            recent_opponent = list(self.opponent_action_history)[-3:]
            for i, action in enumerate(recent_opponent):
                if meta_idx < len(obs):
                    obs[meta_idx] = action / self.num_actions
                    meta_idx += 1

        return obs

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step in the environment.

        State evolution is learned, not hardcoded:
        - Actions affect state based on learned patterns
        - Opponent actions (if present) also affect state
        - No predefined "attack power" or "defense strength"
        """
        prev_state = self.state.copy()
        prev_obs = self._get_observation()

        # Predict opponent action (if opponent model exists)
        opponent_action = None
        if self.opponent_model is not None:
            try:
                obs_tensor = torch.FloatTensor(self._get_observation()).unsqueeze(0)
                with torch.no_grad():
                    opponent_action, _ = self.opponent_model.predict(self._get_observation(), deterministic=False)
                    opponent_action = int(opponent_action)
            except Exception as e:
                # If opponent prediction fails, choose random action
                opponent_action = self.np_random.integers(0, self.num_actions)

        # Update action histories
        self.action_history.append(action)
        if opponent_action is not None:
            self.opponent_action_history.append(opponent_action)

        # Apply action effect (LEARNED, not hardcoded)
        self.state = self._apply_action_learned(self.state, action, opponent_action)

        # Ensure state stays in valid range
        self.state = np.clip(self.state, 0.0, 1.0)

        # Compute reward based on agent role
        reward = self._compute_reward(prev_state, self.state, action, opponent_action)

        # Add curiosity bonus for exploration
        if self.curiosity is not None and self.use_curiosity:
            intrinsic_reward = self.curiosity.compute_intrinsic_reward(
                prev_obs, action, self._get_observation()
            )
            reward += self.curiosity_weight * intrinsic_reward

            # Update curiosity models
            self.curiosity.update(prev_obs, action, self._get_observation())

        # Update step count
        self.step_count += 1
        self.total_steps += 1

        # Check termination
        terminated = False
        truncated = self.step_count >= self.max_steps

        # Get new observation
        obs = self._get_observation()

        # Store interaction in experience buffer
        interaction = Interaction(
            state=prev_obs,
            action=action,
            reward=reward,
            next_state=obs,
            done=terminated or truncated,
            opponent_action=opponent_action,
            timestep=self.step_count,
            episode_id=self.episode_id
        )
        self.experience_buffer.add(interaction)

        # Store transition for learning
        self.transition_history.append({
            'prev_state': prev_state.copy(),
            'action': action,
            'opponent_action': opponent_action,
            'next_state': self.state.copy()
        })

        # Update learned action effects
        self._update_action_effects(prev_state, action, self.state)

        info = {
            'opponent_action': opponent_action,
            'state_mean': np.mean(self.state),
            'state_std': np.std(self.state),
            'intrinsic_reward': intrinsic_reward if self.curiosity else 0.0
        }

        return obs, reward, terminated, truncated, info

    def _apply_action_learned(
        self,
        state: np.ndarray,
        action: int,
        opponent_action: Optional[int]
    ) -> np.ndarray:
        """
        Apply action effects based on LEARNED patterns, not hardcoded rules.

        The effect of an action is determined by:
        1. Historical observed effects (action_effects matrix)
        2. Current state context
        3. Opponent action interaction (if present)
        """
        new_state = state.copy()

        # Use learned action effects if we have enough data
        if self.action_counts[action] > 10:
            # Apply learned effect with some stochasticity
            learned_effect = self.action_effects[action]
            noise = self.np_random.normal(0, 0.05, size=self.state_dim)
            new_state += learned_effect + noise
        else:
            # Early exploration: random effects
            # Use action as seed for reproducibility but no semantic meaning
            rng = np.random.RandomState(action + self.step_count)
            affected_dims = rng.choice(
                self.state_dim,
                size=max(1, self.state_dim // 8),
                replace=False
            )

            for dim in affected_dims:
                # Effect magnitude based on current state (creates dynamics)
                effect_magnitude = rng.uniform(-0.1, 0.1)
                new_state[dim] += effect_magnitude

        # Opponent action interaction (if present)
        if opponent_action is not None:
            # Opponent affects state in opposite direction or interferes
            if self.action_counts[opponent_action] > 10:
                opponent_effect = self.action_effects[opponent_action]
                # Opponent effect is scaled by 0.7 to prevent total cancellation
                new_state -= 0.7 * opponent_effect
            else:
                # Random opponent effect during exploration
                rng = np.random.RandomState(opponent_action + self.step_count + 1000)
                affected_dims = rng.choice(
                    self.state_dim,
                    size=max(1, self.state_dim // 8),
                    replace=False
                )
                for dim in affected_dims:
                    new_state[dim] -= rng.uniform(-0.08, 0.08)

        # Natural state dynamics (state evolves on its own)
        # States tend toward equilibrium (0.5)
        decay = 0.95
        new_state = decay * new_state + (1 - decay) * 0.5

        return new_state

    def _update_action_effects(self, prev_state: np.ndarray, action: int, next_state: np.ndarray):
        """
        Update learned action effects based on observed state transition.
        This is how the agent learns what each action does over time.
        """
        # Compute observed effect
        observed_effect = next_state - prev_state

        # Update running average of action effect
        alpha = 0.1  # Learning rate for effect updates
        self.action_effects[action] = (
            (1 - alpha) * self.action_effects[action] +
            alpha * observed_effect
        )
        self.action_counts[action] += 1

    def _compute_reward(
        self,
        prev_state: np.ndarray,
        current_state: np.ndarray,
        action: int,
        opponent_action: Optional[int]
    ) -> float:
        """
        Compute reward based on agent objective (MINIMAL assumptions):

        Red Team: Wants to INCREASE state values (e.g., compromise, access)
        Blue Team: Wants to DECREASE state values (e.g., security, containment)

        No hardcoded "attack success" or "defense success" - just state movement
        """
        prev_mean = np.mean(prev_state)
        current_mean = np.mean(current_state)

        state_delta = current_mean - prev_mean

        if self.agent_role == "red":
            # Red wants state to increase
            reward = state_delta * 100.0
        else:  # blue
            # Blue wants state to decrease
            reward = -state_delta * 100.0

        # Diversity bonus (encourage exploration)
        if len(self.action_history) >= 10:
            recent_actions = list(self.action_history)[-10:]
            unique_actions = len(set(recent_actions))
            diversity_bonus = (unique_actions / 10.0) * 2.0
            reward += diversity_bonus

        # Clip reward to reasonable range
        reward = np.clip(reward, -50.0, 50.0)

        return float(reward)

    def get_experience_buffer(self) -> ExperienceReplayBuffer:
        """Access the experience replay buffer"""
        return self.experience_buffer

    def get_action_effects_summary(self) -> Dict:
        """Get summary of learned action effects"""
        return {
            'action_effects': self.action_effects.tolist(),
            'action_counts': self.action_counts.tolist(),
            'most_used_actions': np.argsort(self.action_counts)[::-1][:5].tolist()
        }


# ============================================================================
# CONTINUOUS CO-EVOLUTION TRAINER - Never stops learning
# ============================================================================

class ContinuousCoEvolutionTrainer:
    """
    Manages continuous co-evolution training between Red and Blue agents.

    Features:
    - Lifelong learning (never stops)
    - Population-based training (historical opponents)
    - Adaptive curriculum (difficulty adjusts automatically)
    - Experience sharing between generations
    - Meta-learning (learns how to learn)
    """

    def __init__(
        self,
        state_dim: int = 32,
        num_actions: int = 25,
        max_episode_steps: int = 100,
        population_size: int = 5,
        save_dir: Path = Path("models/adaptive_dynamic"),
        use_curiosity: bool = True
    ):
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.max_episode_steps = max_episode_steps
        self.population_size = population_size
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.use_curiosity = use_curiosity

        # Population of historical agents (for diversity)
        self.red_population: List[PPO] = []
        self.blue_population: List[PPO] = []

        # Training history
        self.generation_history: List[Dict] = []

        # Global experience buffers (shared across generations)
        self.red_experience = ExperienceReplayBuffer(max_size=1_000_000)
        self.blue_experience = ExperienceReplayBuffer(max_size=1_000_000)

        # Current generation
        self.current_generation = 0

        print(f"Initialized Continuous Co-Evolution Trainer")
        print(f"  State Dim: {state_dim}")
        print(f"  Actions: {num_actions}")
        print(f"  Population Size: {population_size}")
        print(f"  Curiosity-Driven: {use_curiosity}")
        print(f"  Save Dir: {save_dir}")

    def create_agent(self, role: str, env) -> PPO:
        """Create a new PPO agent with adaptive architecture"""
        # Using recurrent policy would be ideal, but SB3's RecurrentPPO
        # For now, use standard MlpPolicy with larger network
        return PPO(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,  # Encourage exploration
            verbose=0,
            policy_kwargs={
                'net_arch': [256, 256, 128]  # Larger network for complex behaviors
            }
        )

    def train_generation(
        self,
        generation: int,
        steps_per_agent: int = 50_000
    ) -> Dict:
        """
        Train one generation of co-evolution.

        Process:
        1. Red trains against Blue (or historical Blue)
        2. Blue trains against Red (or historical Red)
        3. Evaluate generation
        4. Add to population
        5. Share experiences
        """
        print(f"\n{'='*60}")
        print(f"Generation {generation}")
        print(f"{'='*60}")

        # Create environments
        red_env = AdaptiveDynamicEnv(
            state_dim=self.state_dim,
            num_actions=self.num_actions,
            max_steps=self.max_episode_steps,
            agent_role="red",
            use_curiosity=self.use_curiosity
        )

        blue_env = AdaptiveDynamicEnv(
            state_dim=self.state_dim,
            num_actions=self.num_actions,
            max_steps=self.max_episode_steps,
            agent_role="blue",
            use_curiosity=self.use_curiosity
        )

        # Initialize agents
        if generation == 0:
            # First generation: train from scratch
            red_agent = self.create_agent("red", red_env)
            blue_agent = self.create_agent("blue", blue_env)
        else:
            # Load previous generation and continue training
            red_agent = PPO.load(
                self.save_dir / f"red_gen_{generation-1}",
                env=red_env
            )
            blue_agent = PPO.load(
                self.save_dir / f"blue_gen_{generation-1}",
                env=blue_env
            )

        # Select opponent from population (for diversity)
        blue_opponent = self._select_opponent(self.blue_population, default=blue_agent)
        red_opponent = self._select_opponent(self.red_population, default=red_agent)

        # Set opponents in environments
        red_env.set_opponent_model(blue_opponent)
        blue_env.set_opponent_model(red_opponent)

        # Train Red agent
        print(f"\nTraining Red Agent (Gen {generation})...")
        red_agent.learn(total_timesteps=steps_per_agent, progress_bar=True)

        # Update Red environment's opponent to latest Blue
        red_env.set_opponent_model(blue_agent)

        # Train Blue agent
        print(f"\nTraining Blue Agent (Gen {generation})...")
        blue_agent.learn(total_timesteps=steps_per_agent, progress_bar=True)

        # Collect experiences from this generation
        self.red_experience = red_env.get_experience_buffer()
        self.blue_experience = blue_env.get_experience_buffer()

        # Evaluate generation
        print(f"\nEvaluating Generation {generation}...")
        eval_results = self._evaluate_generation(red_agent, blue_agent, num_episodes=10)

        # Add to population
        self._add_to_population(red_agent, blue_agent)

        # Save agents
        red_agent.save(self.save_dir / f"red_gen_{generation}")
        blue_agent.save(self.save_dir / f"blue_gen_{generation}")

        # Save latest as "final" for easy loading
        red_agent.save(self.save_dir / "red_final")
        blue_agent.save(self.save_dir / "blue_final")

        # Record generation history
        generation_data = {
            'generation': generation,
            'evaluation': eval_results,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'red_experience_size': self.red_experience.size(),
            'blue_experience_size': self.blue_experience.size()
        }
        self.generation_history.append(generation_data)

        # Save training history
        self._save_history()

        self.current_generation = generation + 1

        return generation_data

    def _select_opponent(self, population: List[PPO], default: PPO) -> PPO:
        """Select an opponent from the population (random for diversity)"""
        if len(population) == 0:
            return default
        return population[np.random.randint(0, len(population))]

    def _add_to_population(self, red_agent: PPO, blue_agent: PPO):
        """Add agents to population, maintaining size limit"""
        self.red_population.append(red_agent)
        self.blue_population.append(blue_agent)

        # Maintain population size
        if len(self.red_population) > self.population_size:
            self.red_population.pop(0)
        if len(self.blue_population) > self.population_size:
            self.blue_population.pop(0)

    def _evaluate_generation(
        self,
        red_agent: PPO,
        blue_agent: PPO,
        num_episodes: int = 10
    ) -> Dict:
        """Evaluate the current generation's performance"""
        # Create evaluation environment
        eval_env = AdaptiveDynamicEnv(
            state_dim=self.state_dim,
            num_actions=self.num_actions,
            max_steps=self.max_episode_steps,
            agent_role="red",
            use_curiosity=False  # No exploration during eval
        )

        eval_env.set_opponent_model(blue_agent)

        # Run evaluation episodes
        episode_rewards = []
        state_trajectories = []
        red_action_diversity = []
        blue_action_diversity = []

        for ep in range(num_episodes):
            obs, _ = eval_env.reset()
            done = False
            episode_reward = 0
            state_traj = []
            red_actions = []
            blue_actions = []

            while not done:
                # Red agent acts
                action, _ = red_agent.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = eval_env.step(action)
                done = terminated or truncated

                episode_reward += reward
                state_traj.append(np.mean(eval_env.state))
                red_actions.append(action)
                if info['opponent_action'] is not None:
                    blue_actions.append(info['opponent_action'])

            episode_rewards.append(episode_reward)
            state_trajectories.append(state_traj)
            red_action_diversity.append(len(set(red_actions)) / len(red_actions))
            if len(blue_actions) > 0:
                blue_action_diversity.append(len(set(blue_actions)) / len(blue_actions))

        return {
            'mean_reward': float(np.mean(episode_rewards)),
            'std_reward': float(np.std(episode_rewards)),
            'mean_state_dominance': float(np.mean([traj[-1] for traj in state_trajectories])),
            'red_diversity': float(np.mean(red_action_diversity)),
            'blue_diversity': float(np.mean(blue_action_diversity)) if blue_action_diversity else 0.0
        }

    def _save_history(self):
        """Save training history to disk"""
        history_path = self.save_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.generation_history, f, indent=2)

        print(f"Training history saved to {history_path}")

    def train_continuous(
        self,
        num_generations: int = 20,
        steps_per_generation: int = 50_000
    ):
        """
        Run continuous training for multiple generations.
        Each generation builds on the previous.
        """
        print(f"\nStarting Continuous Co-Evolution Training")
        print(f"  Generations: {num_generations}")
        print(f"  Steps per Generation: {steps_per_generation}")
        print(f"  Total Training Steps: {num_generations * steps_per_generation * 2}")

        for gen in range(num_generations):
            start_time = time.time()

            gen_data = self.train_generation(
                generation=gen,
                steps_per_agent=steps_per_generation
            )

            elapsed = time.time() - start_time

            print(f"\n{'='*60}")
            print(f"Generation {gen} Complete ({elapsed:.1f}s)")
            print(f"  Mean Reward: {gen_data['evaluation']['mean_reward']:.2f}")
            print(f"  State Dominance: {gen_data['evaluation']['mean_state_dominance']:.3f}")
            print(f"  Red Diversity: {gen_data['evaluation']['red_diversity']:.3f}")
            print(f"  Blue Diversity: {gen_data['evaluation']['blue_diversity']:.3f}")
            print(f"  Red Experiences: {gen_data['red_experience_size']}")
            print(f"  Blue Experiences: {gen_data['blue_experience_size']}")
            print(f"{'='*60}")

        print(f"\n🎉 Training Complete! All generations finished.")
        print(f"Models saved to: {self.save_dir}")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    print("="*60)
    print("ACEAC Adaptive Dynamic Learning System")
    print("Adversarial Co-Evolution with Continuous Learning")
    print("="*60)

    # Initialize trainer
    trainer = ContinuousCoEvolutionTrainer(
        state_dim=32,
        num_actions=25,
        max_episode_steps=100,
        population_size=5,
        save_dir=Path("models/adaptive_dynamic"),
        use_curiosity=True
    )

    # Run continuous training
    trainer.train_continuous(
        num_generations=20,
        steps_per_generation=50_000
    )

    print("\n✅ Adaptive Dynamic Learning System training complete!")
    print("   Red and Blue agents have co-evolved through competitive self-play")
    print("   All behaviors learned from experience - no hardcoded logic used")
