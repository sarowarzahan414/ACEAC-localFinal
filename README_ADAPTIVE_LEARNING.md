# Adaptive Dynamic Learning System

## Overview

The **Adaptive Dynamic Learning System** is a fully autonomous reinforcement learning framework where Red Team and Blue Team agents learn cybersecurity strategies entirely from experience, with **NO predefined logic or static rules**. The system implements competitive co-evolution through adversarial self-play, enabling the emergence of sophisticated attack and defense strategies.

## Key Features

### 🧠 **Pure Learning Architecture**
- **No hardcoded attack/defense semantics** - Actions have no predetermined meanings
- **Learned action effects** - Agents discover what each action does through experience
- **Emergent behaviors** - Strategies emerge naturally from competitive pressure
- **Dynamic state evolution** - Environment dynamics learned, not programmed

### 🔄 **Continuous Learning**
- **Experience replay buffer** - All past interactions stored and learned from
- **Lifelong learning** - Never stops improving
- **Population-based training** - Historical opponents prevent cyclic dominance
- **Meta-learning capabilities** - System learns how to learn more effectively

### 🎯 **Curiosity-Driven Exploration**
- **Intrinsic motivation** - Agents rewarded for discovering novel situations
- **Forward/inverse models** - Predict consequences of actions
- **Adaptive exploration** - Balances exploitation and exploration dynamically

### 🧬 **Recurrent Architectures**
- **LSTM policies** - Remember long-term temporal patterns
- **Attention mechanisms** - Focus on important observations
- **Memory-augmented networks** - External memory for complex strategies
- **Hierarchical processing** - Multi-timescale learning (reactive + strategic)

### 📚 **Adaptive Curriculum**
- **Self-paced learning** - Agents control their own difficulty progression
- **Automatic stage transitions** - Advances when ready, regresses when struggling
- **Independent curricula** - Red and Blue progress at their own pace
- **No manual tuning** - Fully autonomous difficulty adjustment

### 📊 **Comprehensive Monitoring**
- **Emergent behavior detection** - Identifies discovered strategies
- **Action pattern analysis** - Reveals tactics and preferences
- **Training progress visualization** - Real-time performance tracking
- **Strategy cataloguing** - Documents all discovered behaviors

## Architecture

### Core Components

```
aceac_adaptive_dynamic_learning.py    - Main adaptive environment & trainer
├── AdaptiveDynamicEnv                - Environment with learned dynamics
├── ExperienceReplayBuffer            - Stores all past interactions
├── CuriosityModule                   - Intrinsic motivation for exploration
└── ContinuousCoEvolutionTrainer      - Manages lifelong co-evolution training

aceac_recurrent_policies.py           - Advanced neural architectures
├── LSTMFeatureExtractor              - LSTM for temporal patterns
├── AttentionFeatureExtractor         - Attention mechanisms
├── MemoryAugmentedNetwork            - External memory
├── HierarchicalFeatureExtractor      - Multi-timescale processing
└── MetaLearningModule                - Learn-to-learn capabilities

aceac_adaptive_curriculum.py          - Self-adjusting difficulty
├── AdaptiveCurriculumManager         - Manages curriculum progression
├── CurriculumStage                   - 5 difficulty levels
└── SelfPacedCoEvolutionTrainer       - Combines curriculum with training

aceac_adaptive_visualization.py       - Monitoring & analysis
├── EmergentBehaviorDetector          - Identifies discovered strategies
├── TrainingProgressMonitor           - Tracks training evolution
└── ActionPatternAnalyzer             - Analyzes action patterns

validate_adaptive_learning.py         - Comprehensive test suite
```

## Quick Start

### 1. Basic Training

```python
from aceac_adaptive_dynamic_learning import ContinuousCoEvolutionTrainer
from pathlib import Path

# Create trainer
trainer = ContinuousCoEvolutionTrainer(
    state_dim=32,
    num_actions=25,
    max_episode_steps=100,
    population_size=5,
    save_dir=Path("models/adaptive_dynamic"),
    use_curiosity=True
)

# Run training
trainer.train_continuous(
    num_generations=20,
    steps_per_generation=50_000
)
```

### 2. With Adaptive Curriculum

```python
from aceac_adaptive_curriculum import SelfPacedCoEvolutionTrainer

# Create self-paced trainer
trainer = SelfPacedCoEvolutionTrainer(
    save_dir=Path("models/adaptive_curriculum")
)

# Training with automatic difficulty adjustment
# (integrate with your training loop)
```

### 3. With Monitoring

```python
from aceac_adaptive_visualization import create_monitoring_system

# Create monitoring system
monitor = create_monitoring_system(save_dir=Path("monitoring"))

# During training, record each generation
monitor.record_generation(
    generation=gen,
    red_metrics={'win_rate': 0.6, 'avg_reward': 45.0, ...},
    blue_metrics={'win_rate': 0.4, 'avg_reward': 35.0, ...},
    red_actions=[[1, 5, 3, ...], ...],
    blue_actions=[[2, 8, 4, ...], ...],
    red_rewards=[[10.0, -5.0, ...], ...],
    blue_rewards=[[-10.0, 5.0, ...], ...]
)

# Generate summary report
print(monitor.generate_summary_report())
```

### 4. Using Recurrent Policies

```python
from stable_baselines3 import PPO
from aceac_recurrent_policies import create_lstm_policy_kwargs
from aceac_adaptive_dynamic_learning import AdaptiveDynamicEnv

# Create environment
env = AdaptiveDynamicEnv(
    state_dim=32,
    num_actions=25,
    agent_role="red",
    use_curiosity=True
)

# Create agent with LSTM policy
policy_kwargs = create_lstm_policy_kwargs(
    lstm_hidden_size=128,
    num_lstm_layers=2,
    features_dim=256
)

agent = PPO(
    "MlpPolicy",
    env,
    policy_kwargs=policy_kwargs,
    learning_rate=3e-4,
    verbose=1
)

# Train
agent.learn(total_timesteps=100_000)
```

## Validation

Run the comprehensive test suite:

```bash
python validate_adaptive_learning.py
```

Tests include:
- ✅ Environment creation and reset
- ✅ Experience replay buffer functionality
- ✅ Curiosity module learning
- ✅ Action effect learning
- ✅ Opponent interaction
- ✅ Recurrent policy architectures
- ✅ Adaptive curriculum progression
- ✅ Integration tests

## Configuration

### Environment Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `state_dim` | 32 | Dimensionality of state space |
| `num_actions` | 25 | Number of available actions |
| `max_steps` | 100 | Maximum steps per episode |
| `agent_role` | "red" | Agent role ("red" or "blue") |
| `use_curiosity` | True | Enable curiosity-driven exploration |
| `curiosity_weight` | 0.1 | Weight of intrinsic rewards |

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_generations` | 20 | Number of training generations |
| `steps_per_generation` | 50,000 | Training steps per generation |
| `population_size` | 5 | Size of opponent population |
| `save_dir` | "models/adaptive_dynamic" | Model save directory |

### Curriculum Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `performance_window` | 100 | Episodes for performance evaluation |
| `advancement_threshold` | 0.75 | Performance needed to advance |
| `regression_threshold` | 0.30 | Performance below which to regress |
| `patience` | 500 | Episodes before forcing progression |

## Curriculum Stages

The adaptive curriculum has 5 stages:

1. **NOVICE** - Basic patterns, simple opponents
   - Episode length: 50
   - Opponent strength: 0.2
   - Advancement: 0.4 mastery

2. **INTERMEDIATE** - More complex interactions
   - Episode length: 75
   - Opponent strength: 0.4
   - Advancement: 0.45 mastery

3. **ADVANCED** - Sophisticated strategies
   - Episode length: 100
   - Opponent strength: 0.6
   - Advancement: 0.5 mastery

4. **EXPERT** - Elite strategies
   - Episode length: 125
   - Opponent strength: 0.8
   - Advancement: 0.55 mastery

5. **MASTER** - Cutting-edge strategies
   - Episode length: 150
   - Opponent strength: 1.0
   - No advancement (final stage)

## Design Philosophy

### Why Pure Learning?

Traditional cybersecurity simulations rely on:
- ❌ Hardcoded attack powers
- ❌ Predefined defense strengths
- ❌ Manual tool mappings
- ❌ Expert-designed reward functions

This system instead uses:
- ✅ **Emergent behaviors** - Strategies discovered through competition
- ✅ **Learned dynamics** - Action effects learned from experience
- ✅ **Minimal assumptions** - Only basic objectives (Red increases state, Blue decreases)
- ✅ **Continuous adaptation** - Never stops learning

### Key Insights

1. **No Semantics Needed** - Actions don't need "meanings" like "SQL Injection". Agents learn what works through trial and error.

2. **Competition Drives Innovation** - Red and Blue pressure each other to improve, leading to arms race dynamics.

3. **Curiosity Enables Discovery** - Intrinsic rewards encourage agents to explore novel strategies.

4. **Memory Enables Strategy** - Recurrent architectures allow agents to execute multi-step plans.

5. **Curriculum Prevents Frustration** - Self-paced learning ensures agents always face appropriate challenges.

## Emergent Behaviors

The system automatically detects and catalogues emergent behaviors:

```python
# Example detected behavior
EmergentBehavior(
    name="Red_Strategy_15",
    description="Repeated action sequence: [5, 12, 3, 18]",
    confidence=0.85,
    first_seen_generation=8,
    action_sequence=[5, 12, 3, 18],
    frequency=23,
    context={
        'avg_reward': 45.3,
        'sequence_length': 4,
        'agent': 'Red'
    }
)
```

## Monitoring Output

During training, you'll see:

```
======================================================================
Generation 10 - Progress Update
======================================================================

📊 Performance Metrics:
  Red Team:
    Win Rate: 58.50%
    Avg Reward: 42.30
    Diversity: 67.20%

  Blue Team:
    Win Rate: 41.50%
    Avg Reward: 38.10
    Diversity: 71.30%

🧬 Emergent Behaviors:
  Red Team: 15 total (3 new)
  Blue Team: 18 total (2 new)

  🆕 New Red Behaviors:
    - Red_Strategy_15: Repeated action sequence: [5, 12, 3, 18]
    - Red_Strategy_16: Repeated action sequence: [2, 7, 15]
    - Red_Strategy_17: Repeated action sequence: [9, 1, 4, 22, 11]

  🆕 New Blue Behaviors:
    - Blue_Strategy_18: Repeated action sequence: [14, 6, 19]
    - Blue_Strategy_19: Repeated action sequence: [3, 21, 8]

⏱️  Elapsed Time: 45.3 minutes
======================================================================
```

## Files Generated

After training, you'll have:

```
models/adaptive_dynamic/
├── red_final.zip                  # Final Red agent
├── blue_final.zip                 # Final Blue agent
├── red_gen_0.zip                  # Generation 0 Red agent
├── blue_gen_0.zip                 # Generation 0 Blue agent
├── ...
└── training_history.json          # Complete training history

monitoring/
├── generation_metrics.json        # Performance per generation
├── red_behaviors.json             # Discovered Red strategies
└── blue_behaviors.json            # Discovered Blue strategies

models/adaptive_curriculum/
├── red_curriculum.json            # Red curriculum state
└── blue_curriculum.json           # Blue curriculum state
```

## Advanced Usage

### Custom Recurrent Architecture

```python
import torch.nn as nn
from aceac_recurrent_policies import BaseFeaturesExtractor

class CustomArchitecture(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256):
        super().__init__(observation_space, features_dim)
        # Your custom architecture here

    def forward(self, observations):
        # Your forward pass
        return features
```

### Custom Curiosity Module

```python
from aceac_adaptive_dynamic_learning import CuriosityModule

class CustomCuriosity(CuriosityModule):
    def compute_intrinsic_reward(self, state, action, next_state):
        # Your custom curiosity computation
        return intrinsic_reward
```

## Performance Tips

1. **Use GPU** - Set `device='cuda'` for faster training
2. **Adjust Batch Size** - Larger batches more stable, smaller faster
3. **Tune Curiosity Weight** - 0.1 works well, adjust if needed
4. **Monitor Diversity** - If agents converge to single strategy, increase exploration
5. **Save Often** - Training can be long, save checkpoints frequently

## Troubleshooting

### Agents Not Learning
- Check curiosity module is enabled
- Verify rewards are non-zero
- Ensure sufficient exploration (diversity > 40%)

### Cyclic Dominance
- Increase population size (> 5)
- Verify opponent selection is random
- Check curriculum is advancing

### Training Too Slow
- Reduce state_dim or num_actions
- Decrease steps_per_generation
- Use smaller neural networks

### Memory Issues
- Reduce experience buffer size
- Decrease population size
- Use smaller batch sizes

## Citation

If you use this system in research, please cite:

```bibtex
@software{aceac_adaptive_learning,
  title={ACEAC Adaptive Dynamic Learning System},
  author={ACEAC Team},
  year={2025},
  description={Fully adaptive reinforcement learning for cybersecurity with adversarial co-evolution}
}
```

## Related Work

- **Dynamic Co-Evolution System** - `aceac_dynamic_coevolution.py`
- **Traditional Co-Evolution** - `aceac_coevolution_FIXED.py`
- **Cyber Kill Chain** - `aceac_v2_cyber_killchain.py`

## License

See main repository LICENSE file.

## Support

For issues, questions, or contributions, see the main ACEAC repository.

---

**🎉 Happy Training! May your agents discover brilliant strategies!**
