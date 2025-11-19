"""
Visualization and Monitoring for Adaptive Learning System
=========================================================

Provides tools to visualize and monitor emergent behaviors in the
adaptive dynamic learning system.

Features:
- Training progress visualization
- Action pattern discovery
- Strategy emergence detection
- Performance evolution tracking
- Curriculum progression charts
- Interaction heatmaps
- Emergent behavior identification
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import Counter
import time


# ============================================================================
# EMERGENT BEHAVIOR DETECTOR
# ============================================================================

@dataclass
class EmergentBehavior:
    """Detected emergent behavior pattern"""
    name: str
    description: str
    confidence: float
    first_seen_generation: int
    action_sequence: List[int]
    frequency: float
    context: Dict

    def to_dict(self) -> Dict:
        return {
            'name': self.name,
            'description': self.description,
            'confidence': self.confidence,
            'first_seen_generation': self.first_seen_generation,
            'action_sequence': self.action_sequence,
            'frequency': self.frequency,
            'context': self.context
        }


class EmergentBehaviorDetector:
    """
    Detects and catalogues emergent behaviors discovered by agents.

    Identifies:
    - Repeated action sequences (strategies)
    - Novel patterns
    - Dominant strategies
    - Counter-strategies
    """

    def __init__(self, min_sequence_length: int = 3, max_sequence_length: int = 10):
        self.min_sequence_length = min_sequence_length
        self.max_sequence_length = max_sequence_length
        self.discovered_behaviors: List[EmergentBehavior] = []
        self.action_sequences: Dict[Tuple, int] = {}

    def analyze_episode(
        self,
        actions: List[int],
        rewards: List[float],
        generation: int,
        agent_name: str
    ) -> List[EmergentBehavior]:
        """
        Analyze an episode to detect emergent behaviors.

        Args:
            actions: List of actions taken
            rewards: List of rewards received
            generation: Current generation number
            agent_name: Name of the agent

        Returns:
            List of detected emergent behaviors
        """
        new_behaviors = []

        # Find frequent action sequences
        for seq_len in range(self.min_sequence_length, min(self.max_sequence_length + 1, len(actions))):
            for i in range(len(actions) - seq_len + 1):
                sequence = tuple(actions[i:i + seq_len])

                # Track sequence
                if sequence not in self.action_sequences:
                    self.action_sequences[sequence] = 0
                self.action_sequences[sequence] += 1

                # Check if this is a significant pattern
                frequency = self.action_sequences[sequence]
                if frequency >= 5 and not self._is_known_behavior(sequence):
                    # Calculate confidence based on reward during sequence
                    seq_rewards = rewards[i:i + seq_len] if i + seq_len <= len(rewards) else []
                    avg_reward = np.mean(seq_rewards) if seq_rewards else 0.0

                    confidence = min(1.0, frequency / 20.0)

                    # Create emergent behavior
                    behavior = EmergentBehavior(
                        name=f"{agent_name}_Strategy_{len(self.discovered_behaviors) + 1}",
                        description=f"Repeated action sequence: {list(sequence)}",
                        confidence=confidence,
                        first_seen_generation=generation,
                        action_sequence=list(sequence),
                        frequency=frequency,
                        context={
                            'avg_reward': float(avg_reward),
                            'sequence_length': seq_len,
                            'agent': agent_name
                        }
                    )

                    self.discovered_behaviors.append(behavior)
                    new_behaviors.append(behavior)

        return new_behaviors

    def _is_known_behavior(self, sequence: Tuple[int, ...]) -> bool:
        """Check if a sequence is already catalogued"""
        for behavior in self.discovered_behaviors:
            if tuple(behavior.action_sequence) == sequence:
                return True
        return False

    def get_dominant_strategies(self, top_k: int = 5) -> List[EmergentBehavior]:
        """Get the most frequent emergent behaviors"""
        sorted_behaviors = sorted(
            self.discovered_behaviors,
            key=lambda b: b.frequency,
            reverse=True
        )
        return sorted_behaviors[:top_k]

    def get_recent_discoveries(self, n_generations: int = 5) -> List[EmergentBehavior]:
        """Get recently discovered behaviors"""
        if not self.discovered_behaviors:
            return []

        latest_gen = max(b.first_seen_generation for b in self.discovered_behaviors)
        cutoff_gen = latest_gen - n_generations

        return [b for b in self.discovered_behaviors if b.first_seen_generation >= cutoff_gen]

    def save(self, filepath: Path):
        """Save discovered behaviors to file"""
        data = {
            'total_behaviors': len(self.discovered_behaviors),
            'behaviors': [b.to_dict() for b in self.discovered_behaviors]
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"Emergent behaviors saved to {filepath}")


# ============================================================================
# TRAINING PROGRESS MONITOR
# ============================================================================

class TrainingProgressMonitor:
    """
    Monitors and reports on training progress in real-time.

    Tracks:
    - Generation performance
    - Learning curves
    - Curriculum progression
    - Emergent behavior discovery
    - Population diversity
    """

    def __init__(self, save_dir: Path = Path("monitoring")):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.generation_metrics: List[Dict] = []
        self.red_behavior_detector = EmergentBehaviorDetector()
        self.blue_behavior_detector = EmergentBehaviorDetector()

        self.start_time = time.time()

    def record_generation(
        self,
        generation: int,
        red_metrics: Dict,
        blue_metrics: Dict,
        red_actions: List[List[int]],
        blue_actions: List[List[int]],
        red_rewards: List[List[float]],
        blue_rewards: List[List[float]],
        additional_info: Optional[Dict] = None
    ):
        """
        Record metrics for a generation.

        Args:
            generation: Generation number
            red_metrics: Red team performance metrics
            blue_metrics: Blue team performance metrics
            red_actions: List of action sequences from Red episodes
            blue_actions: List of action sequences from Blue episodes
            red_rewards: List of reward sequences from Red episodes
            blue_rewards: List of reward sequences from Blue episodes
            additional_info: Optional additional information
        """
        # Analyze emergent behaviors
        red_behaviors = []
        for actions, rewards in zip(red_actions, red_rewards):
            behaviors = self.red_behavior_detector.analyze_episode(
                actions, rewards, generation, "Red"
            )
            red_behaviors.extend(behaviors)

        blue_behaviors = []
        for actions, rewards in zip(blue_actions, blue_rewards):
            behaviors = self.blue_behavior_detector.analyze_episode(
                actions, rewards, generation, "Blue"
            )
            blue_behaviors.extend(behaviors)

        # Record generation data
        gen_data = {
            'generation': generation,
            'timestamp': time.time() - self.start_time,
            'red_metrics': red_metrics,
            'blue_metrics': blue_metrics,
            'red_new_behaviors': len(red_behaviors),
            'blue_new_behaviors': len(blue_behaviors),
            'red_total_behaviors': len(self.red_behavior_detector.discovered_behaviors),
            'blue_total_behaviors': len(self.blue_behavior_detector.discovered_behaviors),
        }

        if additional_info:
            gen_data.update(additional_info)

        self.generation_metrics.append(gen_data)

        # Print progress
        self._print_progress(generation, gen_data, red_behaviors, blue_behaviors)

        # Save periodically
        if generation % 5 == 0:
            self.save_progress()

    def _print_progress(
        self,
        generation: int,
        metrics: Dict,
        red_behaviors: List[EmergentBehavior],
        blue_behaviors: List[EmergentBehavior]
    ):
        """Print formatted progress update"""
        print(f"\n{'='*70}")
        print(f"Generation {generation} - Progress Update")
        print(f"{'='*70}")

        print(f"\n📊 Performance Metrics:")
        print(f"  Red Team:")
        print(f"    Win Rate: {metrics['red_metrics'].get('win_rate', 0):.2%}")
        print(f"    Avg Reward: {metrics['red_metrics'].get('avg_reward', 0):.2f}")
        print(f"    Diversity: {metrics['red_metrics'].get('action_diversity', 0):.2%}")

        print(f"\n  Blue Team:")
        print(f"    Win Rate: {metrics['blue_metrics'].get('win_rate', 0):.2%}")
        print(f"    Avg Reward: {metrics['blue_metrics'].get('avg_reward', 0):.2f}")
        print(f"    Diversity: {metrics['blue_metrics'].get('action_diversity', 0):.2%}")

        print(f"\n🧬 Emergent Behaviors:")
        print(f"  Red Team: {metrics['red_total_behaviors']} total ({len(red_behaviors)} new)")
        print(f"  Blue Team: {metrics['blue_total_behaviors']} total ({len(blue_behaviors)} new)")

        if red_behaviors:
            print(f"\n  🆕 New Red Behaviors:")
            for behavior in red_behaviors[:3]:  # Show first 3
                print(f"    - {behavior.name}: {behavior.description}")

        if blue_behaviors:
            print(f"\n  🆕 New Blue Behaviors:")
            for behavior in blue_behaviors[:3]:
                print(f"    - {behavior.name}: {behavior.description}")

        elapsed = time.time() - self.start_time
        print(f"\n⏱️  Elapsed Time: {elapsed/60:.1f} minutes")
        print(f"{'='*70}\n")

    def save_progress(self):
        """Save all monitoring data"""
        # Save generation metrics
        metrics_path = self.save_dir / "generation_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(self.generation_metrics, f, indent=2)

        # Save emergent behaviors
        self.red_behavior_detector.save(self.save_dir / "red_behaviors.json")
        self.blue_behavior_detector.save(self.save_dir / "blue_behaviors.json")

        print(f"Progress saved to {self.save_dir}")

    def generate_summary_report(self) -> str:
        """Generate a comprehensive summary report"""
        if not self.generation_metrics:
            return "No training data available."

        latest = self.generation_metrics[-1]
        total_gens = len(self.generation_metrics)

        # Calculate trends
        red_rewards = [m['red_metrics'].get('avg_reward', 0) for m in self.generation_metrics]
        blue_rewards = [m['blue_metrics'].get('avg_reward', 0) for m in self.generation_metrics]

        red_trend = "↗" if len(red_rewards) > 1 and red_rewards[-1] > red_rewards[0] else "→"
        blue_trend = "↗" if len(blue_rewards) > 1 and blue_rewards[-1] > blue_rewards[0] else "→"

        # Get dominant strategies
        red_dominant = self.red_behavior_detector.get_dominant_strategies(top_k=3)
        blue_dominant = self.blue_behavior_detector.get_dominant_strategies(top_k=3)

        report = f"""
{'='*70}
ADAPTIVE DYNAMIC LEARNING - TRAINING SUMMARY
{'='*70}

📈 Training Overview:
  Total Generations: {total_gens}
  Training Time: {(time.time() - self.start_time)/60:.1f} minutes

🎯 Final Performance:
  Red Team:
    Win Rate: {latest['red_metrics'].get('win_rate', 0):.2%} {red_trend}
    Avg Reward: {latest['red_metrics'].get('avg_reward', 0):.2f}
    Diversity: {latest['red_metrics'].get('action_diversity', 0):.2%}

  Blue Team:
    Win Rate: {latest['blue_metrics'].get('win_rate', 0):.2%} {blue_trend}
    Avg Reward: {latest['blue_metrics'].get('avg_reward', 0):.2f}
    Diversity: {latest['blue_metrics'].get('action_diversity', 0):.2%}

🧬 Emergent Behaviors Discovered:
  Red Team: {latest['red_total_behaviors']} strategies
  Blue Team: {latest['blue_total_behaviors']} strategies

🏆 Dominant Red Strategies:
"""
        for i, behavior in enumerate(red_dominant, 1):
            report += f"  {i}. {behavior.name} (frequency: {behavior.frequency})\n"
            report += f"     {behavior.description}\n"

        report += f"\n🛡️ Dominant Blue Strategies:\n"
        for i, behavior in enumerate(blue_dominant, 1):
            report += f"  {i}. {behavior.name} (frequency: {behavior.frequency})\n"
            report += f"     {behavior.description}\n"

        report += f"\n{'='*70}\n"

        return report

    def get_action_distribution(self, agent: str = "red") -> Dict[int, int]:
        """Get distribution of actions used by an agent"""
        if agent == "red":
            behaviors = self.red_behavior_detector.discovered_behaviors
        else:
            behaviors = self.blue_behavior_detector.discovered_behaviors

        action_counts = Counter()
        for behavior in behaviors:
            for action in behavior.action_sequence:
                action_counts[action] += behavior.frequency

        return dict(action_counts)


# ============================================================================
# ACTION PATTERN ANALYZER
# ============================================================================

class ActionPatternAnalyzer:
    """
    Analyzes action patterns to identify strategies and tactics.

    Provides insights into:
    - Action transition probabilities
    - Action co-occurrence
    - Temporal patterns
    - State-action relationships
    """

    def __init__(self, num_actions: int):
        self.num_actions = num_actions
        self.transition_matrix = np.zeros((num_actions, num_actions))
        self.action_counts = np.zeros(num_actions)

    def update(self, actions: List[int]):
        """Update statistics with new action sequence"""
        for action in actions:
            if 0 <= action < self.num_actions:
                self.action_counts[action] += 1

        # Update transitions
        for i in range(len(actions) - 1):
            curr_action = actions[i]
            next_action = actions[i + 1]
            if 0 <= curr_action < self.num_actions and 0 <= next_action < self.num_actions:
                self.transition_matrix[curr_action, next_action] += 1

    def get_transition_probabilities(self) -> np.ndarray:
        """Get action transition probability matrix"""
        # Normalize each row
        row_sums = self.transition_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        return self.transition_matrix / row_sums

    def get_most_likely_next_action(self, current_action: int) -> Tuple[int, float]:
        """Get most likely next action given current action"""
        if not (0 <= current_action < self.num_actions):
            return 0, 0.0

        probs = self.get_transition_probabilities()
        next_action = np.argmax(probs[current_action])
        probability = probs[current_action, next_action]

        return int(next_action), float(probability)

    def get_action_preferences(self) -> List[Tuple[int, float]]:
        """Get actions sorted by usage frequency"""
        total = self.action_counts.sum()
        if total == 0:
            return []

        preferences = [
            (action, count / total)
            for action, count in enumerate(self.action_counts)
        ]
        return sorted(preferences, key=lambda x: x[1], reverse=True)

    def get_entropy(self) -> float:
        """Calculate entropy of action distribution (measure of diversity)"""
        total = self.action_counts.sum()
        if total == 0:
            return 0.0

        probs = self.action_counts / total
        probs = probs[probs > 0]  # Remove zeros
        entropy = -np.sum(probs * np.log2(probs))

        return float(entropy)


# ============================================================================
# MAIN MONITORING INTERFACE
# ============================================================================

def create_monitoring_system(save_dir: Path = Path("monitoring")) -> TrainingProgressMonitor:
    """
    Create a complete monitoring system for adaptive learning.

    Args:
        save_dir: Directory to save monitoring data

    Returns:
        TrainingProgressMonitor instance
    """
    monitor = TrainingProgressMonitor(save_dir=save_dir)
    print(f"Monitoring system initialized")
    print(f"  Save directory: {save_dir}")
    return monitor


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("Adaptive Learning Visualization & Monitoring System")
    print("="*70)

    # Create monitoring system
    monitor = create_monitoring_system()

    # Simulate training data
    print("\nSimulating training progress...")
    for gen in range(5):
        # Simulate metrics
        red_metrics = {
            'win_rate': 0.3 + gen * 0.1 + np.random.rand() * 0.1,
            'avg_reward': 20.0 + gen * 10.0 + np.random.randn() * 5,
            'action_diversity': 0.4 + gen * 0.05
        }

        blue_metrics = {
            'win_rate': 0.7 - gen * 0.1 + np.random.rand() * 0.1,
            'avg_reward': 60.0 - gen * 5.0 + np.random.randn() * 5,
            'action_diversity': 0.5 + gen * 0.03
        }

        # Simulate action sequences
        red_actions = [
            [np.random.randint(0, 10) for _ in range(50)]
            for _ in range(10)
        ]
        blue_actions = [
            [np.random.randint(0, 10) for _ in range(50)]
            for _ in range(10)
        ]

        # Simulate rewards
        red_rewards = [
            [np.random.randn() * 10 for _ in range(50)]
            for _ in range(10)
        ]
        blue_rewards = [
            [np.random.randn() * 10 for _ in range(50)]
            for _ in range(10)
        ]

        # Record generation
        monitor.record_generation(
            generation=gen,
            red_metrics=red_metrics,
            blue_metrics=blue_metrics,
            red_actions=red_actions,
            blue_actions=blue_actions,
            red_rewards=red_rewards,
            blue_rewards=blue_rewards
        )

    # Generate summary report
    print("\n" + monitor.generate_summary_report())

    print("\n✅ Monitoring system demonstration complete!")
