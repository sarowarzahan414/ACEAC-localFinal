"""
Adaptive Curriculum Learning System
===================================

Implements self-adjusting curriculum where agents dynamically control
their training difficulty based on performance and learning progress.

Key Features:
- Automatic difficulty adjustment based on win rates and performance
- Multi-stage curriculum (novice -> intermediate -> expert -> master)
- Self-paced learning (agents control progression speed)
- Adaptive opponent strength selection
- Dynamic environment complexity scaling
- No manual tuning required - fully autonomous

Philosophy:
- Agents should face challenges they can learn from, not be overwhelmed
- Difficulty increases when agents master current level
- Difficulty decreases when agents struggle too much
- System finds optimal "zone of proximal development" automatically
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path


# ============================================================================
# CURRICULUM STAGES
# ============================================================================

class CurriculumStage(Enum):
    """Stages of learning difficulty"""
    NOVICE = 1        # Basic patterns, simple opponents
    INTERMEDIATE = 2  # More complex interactions, moderate opponents
    ADVANCED = 3      # Sophisticated strategies, strong opponents
    EXPERT = 4        # Elite strategies, expert opponents
    MASTER = 5        # Cutting-edge strategies, master opponents


@dataclass
class StageConfig:
    """Configuration for a curriculum stage"""
    stage: CurriculumStage
    episode_length: int          # Max steps per episode
    opponent_strength: float     # 0.0 (weakest) to 1.0 (strongest)
    state_complexity: float      # Complexity of state dynamics
    action_noise: float          # Exploration noise level
    reward_scale: float          # Reward scaling factor
    min_performance: float       # Min performance to advance
    max_performance: float       # Max performance before forced advance

    def to_dict(self) -> Dict:
        return {
            'stage': self.stage.name,
            'episode_length': self.episode_length,
            'opponent_strength': self.opponent_strength,
            'state_complexity': self.state_complexity,
            'action_noise': self.action_noise,
            'reward_scale': self.reward_scale,
            'min_performance': self.min_performance,
            'max_performance': self.max_performance
        }


# ============================================================================
# PERFORMANCE METRICS
# ============================================================================

@dataclass
class PerformanceMetrics:
    """Tracks agent performance for curriculum decisions"""
    win_rate: float              # Fraction of episodes won
    avg_reward: float            # Average episode reward
    action_diversity: float      # Unique actions / total actions
    learning_rate: float         # Rate of performance improvement
    consistency: float           # Std dev of recent performance
    mastery_score: float         # Overall mastery (0-1)

    def to_dict(self) -> Dict:
        return {
            'win_rate': self.win_rate,
            'avg_reward': self.avg_reward,
            'action_diversity': self.action_diversity,
            'learning_rate': self.learning_rate,
            'consistency': self.consistency,
            'mastery_score': self.mastery_score
        }


# ============================================================================
# ADAPTIVE CURRICULUM MANAGER
# ============================================================================

class AdaptiveCurriculumManager:
    """
    Manages curriculum progression based on agent performance.

    Automatically adjusts difficulty to maintain optimal learning conditions.
    No manual intervention required.
    """

    def __init__(
        self,
        agent_name: str = "agent",
        initial_stage: CurriculumStage = CurriculumStage.NOVICE,
        performance_window: int = 100,  # Episodes to evaluate performance
        patience: int = 500,            # Episodes before forcing progression
        advancement_threshold: float = 0.75,  # Performance needed to advance
        regression_threshold: float = 0.30,   # Performance below which to regress
    ):
        self.agent_name = agent_name
        self.current_stage = initial_stage
        self.performance_window = performance_window
        self.patience = patience
        self.advancement_threshold = advancement_threshold
        self.regression_threshold = regression_threshold

        # Performance tracking
        self.episode_history: List[Dict] = []
        self.stage_history: List[Dict] = []
        self.episodes_in_current_stage = 0

        # Stage configurations
        self.stage_configs = self._create_stage_configs()

        print(f"Adaptive Curriculum Manager initialized for {agent_name}")
        print(f"  Initial Stage: {initial_stage.name}")
        print(f"  Performance Window: {performance_window} episodes")
        print(f"  Advancement Threshold: {advancement_threshold}")
        print(f"  Regression Threshold: {regression_threshold}")

    def _create_stage_configs(self) -> Dict[CurriculumStage, StageConfig]:
        """Create configuration for each curriculum stage"""
        configs = {
            CurriculumStage.NOVICE: StageConfig(
                stage=CurriculumStage.NOVICE,
                episode_length=50,
                opponent_strength=0.2,
                state_complexity=0.3,
                action_noise=0.3,
                reward_scale=1.0,
                min_performance=0.4,
                max_performance=0.8
            ),
            CurriculumStage.INTERMEDIATE: StageConfig(
                stage=CurriculumStage.INTERMEDIATE,
                episode_length=75,
                opponent_strength=0.4,
                state_complexity=0.5,
                action_noise=0.2,
                reward_scale=1.0,
                min_performance=0.45,
                max_performance=0.8
            ),
            CurriculumStage.ADVANCED: StageConfig(
                stage=CurriculumStage.ADVANCED,
                episode_length=100,
                opponent_strength=0.6,
                state_complexity=0.7,
                action_noise=0.15,
                reward_scale=1.0,
                min_performance=0.5,
                max_performance=0.85
            ),
            CurriculumStage.EXPERT: StageConfig(
                stage=CurriculumStage.EXPERT,
                episode_length=125,
                opponent_strength=0.8,
                state_complexity=0.85,
                action_noise=0.1,
                reward_scale=1.0,
                min_performance=0.55,
                max_performance=0.9
            ),
            CurriculumStage.MASTER: StageConfig(
                stage=CurriculumStage.MASTER,
                episode_length=150,
                opponent_strength=1.0,
                state_complexity=1.0,
                action_noise=0.05,
                reward_scale=1.0,
                min_performance=0.6,
                max_performance=1.0
            )
        }
        return configs

    def record_episode(
        self,
        reward: float,
        steps: int,
        won: bool,
        actions: List[int],
        additional_metrics: Optional[Dict] = None
    ):
        """
        Record episode results for curriculum decision-making.

        Args:
            reward: Total episode reward
            steps: Number of steps taken
            won: Whether the episode was won
            actions: List of actions taken
            additional_metrics: Optional additional metrics
        """
        episode_data = {
            'episode': len(self.episode_history),
            'stage': self.current_stage.name,
            'reward': reward,
            'steps': steps,
            'won': won,
            'action_diversity': len(set(actions)) / len(actions) if actions else 0.0
        }

        if additional_metrics:
            episode_data.update(additional_metrics)

        self.episode_history.append(episode_data)
        self.episodes_in_current_stage += 1

    def should_advance(self) -> bool:
        """
        Determine if agent should advance to next difficulty level.

        Returns:
            True if agent is ready to advance
        """
        if len(self.episode_history) < self.performance_window:
            return False

        # Already at highest level
        if self.current_stage == CurriculumStage.MASTER:
            return False

        # Get recent performance
        metrics = self.compute_performance_metrics()

        # Get current stage config
        config = self.stage_configs[self.current_stage]

        # Check advancement criteria
        # 1. Performance exceeds threshold
        performance_ready = metrics.mastery_score >= self.advancement_threshold

        # 2. Performance is very high (forced advancement to prevent plateau)
        performance_too_high = metrics.win_rate >= config.max_performance

        # 3. Been in stage long enough and learning has plateaued
        time_ready = (
            self.episodes_in_current_stage >= self.patience and
            metrics.learning_rate < 0.01  # Minimal improvement
        )

        return performance_ready or performance_too_high or time_ready

    def should_regress(self) -> bool:
        """
        Determine if agent should regress to easier difficulty.

        Returns:
            True if agent is struggling too much
        """
        if len(self.episode_history) < self.performance_window:
            return False

        # Already at lowest level
        if self.current_stage == CurriculumStage.NOVICE:
            return False

        # Get recent performance
        metrics = self.compute_performance_metrics()

        # Regress if performance is too low
        return metrics.mastery_score < self.regression_threshold

    def advance_stage(self):
        """Advance to next curriculum stage"""
        current_value = self.current_stage.value
        if current_value < len(CurriculumStage):
            new_stage = CurriculumStage(current_value + 1)
            self._transition_to_stage(new_stage, reason="advancement")

    def regress_stage(self):
        """Regress to previous curriculum stage"""
        current_value = self.current_stage.value
        if current_value > 1:
            new_stage = CurriculumStage(current_value - 1)
            self._transition_to_stage(new_stage, reason="regression")

    def _transition_to_stage(self, new_stage: CurriculumStage, reason: str):
        """Transition to a new curriculum stage"""
        old_stage = self.current_stage

        # Record stage transition
        transition_data = {
            'from_stage': old_stage.name,
            'to_stage': new_stage.name,
            'reason': reason,
            'episode': len(self.episode_history),
            'episodes_in_stage': self.episodes_in_current_stage,
            'performance': self.compute_performance_metrics().to_dict()
        }
        self.stage_history.append(transition_data)

        # Update current stage
        self.current_stage = new_stage
        self.episodes_in_current_stage = 0

        print(f"\n{'='*60}")
        print(f"📚 Curriculum Transition: {old_stage.name} → {new_stage.name}")
        print(f"   Reason: {reason}")
        print(f"   Episode: {len(self.episode_history)}")
        print(f"{'='*60}\n")

    def compute_performance_metrics(self) -> PerformanceMetrics:
        """
        Compute performance metrics from recent episodes.

        Returns:
            Performance metrics
        """
        if len(self.episode_history) == 0:
            return PerformanceMetrics(
                win_rate=0.0,
                avg_reward=0.0,
                action_diversity=0.0,
                learning_rate=0.0,
                consistency=0.0,
                mastery_score=0.0
            )

        # Get recent episodes
        recent = self.episode_history[-self.performance_window:]

        # Win rate
        win_rate = np.mean([ep['won'] for ep in recent])

        # Average reward
        avg_reward = np.mean([ep['reward'] for ep in recent])

        # Action diversity
        avg_diversity = np.mean([ep['action_diversity'] for ep in recent])

        # Learning rate (trend in rewards)
        if len(recent) >= 20:
            first_half_reward = np.mean([ep['reward'] for ep in recent[:len(recent)//2]])
            second_half_reward = np.mean([ep['reward'] for ep in recent[len(recent)//2:]])
            learning_rate = max(0.0, (second_half_reward - first_half_reward) / (abs(first_half_reward) + 1e-6))
        else:
            learning_rate = 0.0

        # Consistency (inverse of std dev, normalized)
        reward_std = np.std([ep['reward'] for ep in recent])
        consistency = 1.0 / (1.0 + reward_std / (abs(avg_reward) + 1e-6))

        # Mastery score (weighted combination)
        mastery_score = (
            0.4 * win_rate +
            0.3 * min(1.0, avg_reward / 100.0) +  # Assuming rewards up to ~100
            0.2 * avg_diversity +
            0.1 * consistency
        )
        mastery_score = np.clip(mastery_score, 0.0, 1.0)

        return PerformanceMetrics(
            win_rate=win_rate,
            avg_reward=avg_reward,
            action_diversity=avg_diversity,
            learning_rate=learning_rate,
            consistency=consistency,
            mastery_score=mastery_score
        )

    def get_current_config(self) -> StageConfig:
        """Get configuration for current curriculum stage"""
        return self.stage_configs[self.current_stage]

    def update(self) -> Optional[str]:
        """
        Update curriculum based on recent performance.

        Should be called periodically (e.g., every episode or every N episodes).

        Returns:
            Action taken ("advance", "regress", or None)
        """
        # Check if advancement is needed
        if self.should_advance():
            self.advance_stage()
            return "advance"

        # Check if regression is needed
        if self.should_regress():
            self.regress_stage()
            return "regress"

        return None

    def get_summary(self) -> Dict:
        """Get summary of curriculum progress"""
        metrics = self.compute_performance_metrics()

        return {
            'agent': self.agent_name,
            'current_stage': self.current_stage.name,
            'total_episodes': len(self.episode_history),
            'episodes_in_current_stage': self.episodes_in_current_stage,
            'stage_transitions': len(self.stage_history),
            'current_performance': metrics.to_dict(),
            'current_config': self.get_current_config().to_dict()
        }

    def save(self, filepath: Path):
        """Save curriculum state to disk"""
        data = {
            'agent_name': self.agent_name,
            'current_stage': self.current_stage.name,
            'episodes_in_current_stage': self.episodes_in_current_stage,
            'episode_history': self.episode_history,
            'stage_history': self.stage_history,
            'summary': self.get_summary()
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"Curriculum state saved to {filepath}")

    def load(self, filepath: Path):
        """Load curriculum state from disk"""
        with open(filepath, 'r') as f:
            data = json.load(f)

        self.agent_name = data['agent_name']
        self.current_stage = CurriculumStage[data['current_stage']]
        self.episodes_in_current_stage = data['episodes_in_current_stage']
        self.episode_history = data['episode_history']
        self.stage_history = data['stage_history']

        print(f"Curriculum state loaded from {filepath}")


# ============================================================================
# SELF-PACED TRAINER - Combines curriculum with training
# ============================================================================

class SelfPacedCoEvolutionTrainer:
    """
    Training system with adaptive curriculum for both Red and Blue agents.

    Each agent has its own curriculum that adapts independently based on
    individual performance.
    """

    def __init__(
        self,
        save_dir: Path = Path("models/adaptive_curriculum")
    ):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Curriculum managers for each agent
        self.red_curriculum = AdaptiveCurriculumManager(
            agent_name="Red Team",
            initial_stage=CurriculumStage.NOVICE
        )

        self.blue_curriculum = AdaptiveCurriculumManager(
            agent_name="Blue Team",
            initial_stage=CurriculumStage.NOVICE
        )

        print(f"\nSelf-Paced Co-Evolution Trainer initialized")
        print(f"  Both agents start at NOVICE level")
        print(f"  Curriculum adapts independently for each agent")

    def should_update_curriculum(self, episode: int, check_frequency: int = 50) -> bool:
        """Check if curriculum should be updated this episode"""
        return episode % check_frequency == 0 and episode > 0

    def update_curriculums(self):
        """Update both curricula based on performance"""
        print(f"\nUpdating Curricula...")

        # Update Red curriculum
        red_action = self.red_curriculum.update()
        if red_action:
            print(f"  Red Team: {red_action}d to {self.red_curriculum.current_stage.name}")

        # Update Blue curriculum
        blue_action = self.blue_curriculum.update()
        if blue_action:
            print(f"  Blue Team: {blue_action}d to {self.blue_curriculum.current_stage.name}")

        # Print current status
        red_metrics = self.red_curriculum.compute_performance_metrics()
        blue_metrics = self.blue_curriculum.compute_performance_metrics()

        print(f"\n  Red Team Status:")
        print(f"    Stage: {self.red_curriculum.current_stage.name}")
        print(f"    Mastery: {red_metrics.mastery_score:.2%}")
        print(f"    Win Rate: {red_metrics.win_rate:.2%}")

        print(f"\n  Blue Team Status:")
        print(f"    Stage: {self.blue_curriculum.current_stage.name}")
        print(f"    Mastery: {blue_metrics.mastery_score:.2%}")
        print(f"    Win Rate: {blue_metrics.win_rate:.2%}")

    def get_training_config(self) -> Dict:
        """Get current training configuration based on curricula"""
        red_config = self.red_curriculum.get_current_config()
        blue_config = self.blue_curriculum.get_current_config()

        # Use the more advanced agent's config for episode length
        # to ensure both agents are challenged
        episode_length = max(red_config.episode_length, blue_config.episode_length)

        return {
            'episode_length': episode_length,
            'red_opponent_strength': red_config.opponent_strength,
            'blue_opponent_strength': blue_config.opponent_strength,
            'red_exploration_noise': red_config.action_noise,
            'blue_exploration_noise': blue_config.action_noise,
        }

    def save_curricula(self):
        """Save both curricula to disk"""
        self.red_curriculum.save(self.save_dir / "red_curriculum.json")
        self.blue_curriculum.save(self.save_dir / "blue_curriculum.json")

    def load_curricula(self):
        """Load both curricula from disk"""
        red_path = self.save_dir / "red_curriculum.json"
        blue_path = self.save_dir / "blue_curriculum.json"

        if red_path.exists():
            self.red_curriculum.load(red_path)
        if blue_path.exists():
            self.blue_curriculum.load(blue_path)


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("="*60)
    print("Adaptive Curriculum Learning System")
    print("="*60)

    # Create curriculum manager
    curriculum = AdaptiveCurriculumManager(
        agent_name="Test Agent",
        initial_stage=CurriculumStage.NOVICE
    )

    print(f"\nSimulating training with adaptive curriculum...")

    # Simulate 1000 episodes of training
    for episode in range(1000):
        # Simulate performance (improving over time)
        base_performance = min(0.9, 0.3 + episode / 1000.0)
        noise = np.random.normal(0, 0.1)
        reward = 50.0 * (base_performance + noise)
        won = np.random.random() < base_performance

        # Simulate actions
        num_actions = 50
        actions = [np.random.randint(0, 10) for _ in range(num_actions)]

        # Record episode
        curriculum.record_episode(
            reward=reward,
            steps=num_actions,
            won=won,
            actions=actions
        )

        # Check for curriculum updates every 50 episodes
        if episode % 50 == 0 and episode > 0:
            curriculum.update()

            if episode % 100 == 0:
                summary = curriculum.get_summary()
                print(f"\nEpisode {episode}:")
                print(f"  Stage: {summary['current_stage']}")
                print(f"  Mastery: {summary['current_performance']['mastery_score']:.2%}")
                print(f"  Transitions: {summary['stage_transitions']}")

    print("\n" + "="*60)
    print("✅ Curriculum simulation complete!")
    print(f"   Final Stage: {curriculum.current_stage.name}")
    print(f"   Total Stage Transitions: {len(curriculum.stage_history)}")
