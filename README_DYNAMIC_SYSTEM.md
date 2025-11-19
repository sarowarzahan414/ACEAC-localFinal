# ACEAC Dynamic Co-Evolution System

**Fully Adaptive Adversarial RL - NO Predefined Logic**

**Author:** @sarowarzahan414
**Date:** 2025-11-18
**Version:** 2.0-DYNAMIC

---

## 🎯 Core Philosophy

> **"The system learns EVERYTHING through experience. No hardcoded rules. No predefined strategies. Pure adaptation."**

---

## 🚀 Quick Start

### Run Dynamic Training
```bash
python aceac_dynamic_coevolution.py
```

**What happens:**
1. Agents start with **random behaviors**
2. Explore action space **without guidance**
3. Learn what works through **trial and error**
4. Develop **emergent strategies** through self-play
5. Co-evolve **novel attack/defense** patterns

**No predetermined logic. All learned.**

---

## 📦 What's Different

### Static System (OLD)
```python
# Actions have predefined meanings
action_0 = "Scan"      # Power: 0.08
action_2 = "Exploit"   # Power: 0.18
action_8 = "Ransomware" # Power: 0.20

# Agent learns: "Use action 8 because it has highest power"
```

### Dynamic System (NEW)
```python
# Actions have NO predefined meanings
action_0 = ???  # Effect learned through experience
action_2 = ???  # Effect learned through experience
action_8 = ???  # Effect learned through experience

# Agent learns: "Action X works against this opponent in this state"
# Different every training run! Emergent discovery!
```

---

## 🧠 How It Works

### 1. No Hardcoded Action Effects

**Static (OLD):**
```python
if action == 12:  # SQL Injection
    attack_power = 0.18
    target = "database"
```

**Dynamic (NEW):**
```python
# Action 12 affects random state dimensions
# Effect magnitude varies randomly
# Effectiveness emerges from rewards
# NO predetermined meaning!
```

---

### 2. Minimal Reward Signal

**Static (OLD):**
```python
reward = (
    attack_power * 100 +
    stealth_bonus * 50 +
    phase_progress * 25 -
    detection_penalty * 75
)
# Complex hardcoded formula
```

**Dynamic (NEW):**
```python
reward = did_state_improve() ? positive : negative
# Simple outcome-based signal
# Agent learns value function from experience
```

---

### 3. Pure State Representation

**Static (OLD):**
```python
obs = [
    network_health,     # Semantic
    red_score,          # Semantic
    blue_score,         # Semantic
    kill_chain_phase,   # Semantic
    detection_level     # Semantic
]
# Each feature has predetermined meaning
```

**Dynamic (NEW):**
```python
obs = raw_state  # Just numbers [0.1, 0.7, 0.3, ...]
# NO semantic meaning
# Agent learns relevant features
```

---

### 4. Emergent Strategy Development

**Static (OLD):**
```python
# Optimal strategy = exploit hardcoded values
Generation 1: Use highest power actions
Generation 10: Still use highest power actions
# No real evolution
```

**Dynamic (NEW):**
```python
# Optimal strategy = discovered through learning
Generation 1: Random exploration
Generation 5: Discovers pattern [3, 7, 15] works
Generation 10: Blue counters, Red finds [22, 1, 9]
Generation 15: Completely new strategies
# True co-evolution!
```

---

## 🔬 Training Process

### Initialization
```python
trainer = DynamicCoEvolutionTrainer(
    num_actions=25,      # 25 available actions
    state_dim=32,        # 32-dimensional state
    population_size=5    # Keep 5 historical opponents
)
```

**Key points:**
- Actions have **NO labels** (no "Scan", "Exploit", etc.)
- State is **pure numeric** (no "health", "score")
- Everything **learned from scratch**

---

### Training Loop

```python
for generation in range(num_generations):
    # 1. Red trains vs Blue (or historical Blue)
    red_env = DynamicCoEvolutionEnv(
        agent_role="red",
        opponent_model=blue_opponent
    )
    red_model.learn(total_timesteps=50000)

    # 2. Blue trains vs Red (current Red)
    blue_env = DynamicCoEvolutionEnv(
        agent_role="blue",
        opponent_model=red_model
    )
    blue_model.learn(total_timesteps=50000)

    # 3. Evaluate and save
    results = evaluate_generation(red_model, blue_model)

    # 4. Add to population (for diversity)
    red_population.append(red_model.copy())
    blue_population.append(blue_model.copy())
```

**What emerges:**
- Red learns **what actions work** against Blue
- Blue learns **how to counter** Red's strategies
- Strategies **continuously evolve**
- No reliance on **predefined logic**

---

## 📊 Expected Behavior

### Early Training (Generations 1-5)
```
Red: Random actions → Some work, some don't
Blue: Random defenses → Some work, some don't
Result: Chaotic, exploratory behavior
```

### Mid Training (Generations 5-15)
```
Red: Discovers effective action sequences
Blue: Learns to counter Red's patterns
Result: Emerging strategies, counter-strategies
```

### Late Training (Generations 15+)
```
Red: Sophisticated multi-step strategies
Blue: Adaptive defense based on Red's patterns
Result: True adversarial co-evolution
```

---

## 🎮 Example Training Run

```bash
$ python aceac_dynamic_coevolution.py

======================================================================
DYNAMIC CO-EVOLUTION TRAINING - NO PREDEFINED LOGIC
======================================================================
Generations: 20
Steps per generation: 50000
Actions: 25 (no predefined meanings!)
State dimensions: 32
======================================================================

Initializing agents...
✓ Agents initialized

======================================================================
GENERATION 1/20
======================================================================

Training Red agent...
[Progress bar] 50000/50000
✓ Red training complete

Training Blue agent...
[Progress bar] 50000/50000
✓ Blue training complete

Evaluating generation 1...
  State dominance: 0.523
  Red diversity: 0.680
  Blue diversity: 0.720

======================================================================
GENERATION 2/20
======================================================================
...

[Continues for 20 generations]

======================================================================
TRAINING COMPLETE
======================================================================

Agents have learned purely through experience!
No predefined logic, no hardcoded strategies.
All behaviors emerged from self-play.

✓ Final models saved:
  models/dynamic/red_final.zip
  models/dynamic/blue_final.zip

✓ Training history saved: models/dynamic/training_history.json
```

---

## 📈 Monitoring Metrics

### State Dominance
```python
state_dominance = mean(state_values)
```
- **0.5**: Balanced (neither winning)
- **> 0.5**: Red dominating
- **< 0.5**: Blue dominating

**Expected:** Oscillates as strategies co-evolve

---

### Action Diversity
```python
diversity = unique_actions / total_actions
```
- **Low (<0.3)**: Strategy collapse (bad!)
- **Medium (0.3-0.7)**: Focused strategy
- **High (>0.7)**: Exploratory behavior

**Expected:** High initially, stabilizes mid-range

---

## 🔍 Analysis Tools

### Analyze Training History
```python
import json

with open('models/dynamic/training_history.json') as f:
    history = json.load(f)

for gen in history['generations']:
    print(f"Gen {gen['generation']}:")
    print(f"  State: {gen['evaluation']['state_dominance']:.3f}")
    print(f"  Red div: {gen['evaluation']['red_diversity']:.3f}")
    print(f"  Blue div: {gen['evaluation']['blue_diversity']:.3f}")
```

---

### Test Against Opponent
```python
from stable_baselines3 import PPO
from aceac_dynamic_coevolution import DynamicCoEvolutionEnv

# Load trained models
red = PPO.load("models/dynamic/red_final.zip")
blue = PPO.load("models/dynamic/blue_final.zip")

# Test Red vs Blue
env = DynamicCoEvolutionEnv(agent_role="red", opponent_model=blue)
obs, _ = env.reset()

for step in range(100):
    action, _ = red.predict(obs, deterministic=True)
    obs, reward, done, _, info = env.step(action)

    print(f"Step {step}: Action={action}, State={info['state_mean']:.3f}")

    if done:
        break
```

---

## 🎯 Best Practices

### 1. Start Small
```python
# Initial experimentation
trainer.train(
    num_generations=10,       # Fewer generations
    steps_per_generation=20000 # Fewer steps
)
```

### 2. Monitor Diversity
```python
# Check for strategy collapse
if diversity < 0.2:
    print("⚠️ Low diversity - add exploration bonus")
```

### 3. Use Population
```python
# Maintain diverse opponents
population_size=5  # At least 5 historical opponents
```

### 4. Trust the Process
```python
# Don't rush to add hardcoded logic!
# Let agents explore, even if initial behavior seems random
# Emergence takes time
```

---

## 🐛 Troubleshooting

### "Agents not learning anything"
**Problem:** Both agents random after many generations

**Solutions:**
- Increase training steps (50k → 100k per generation)
- Check reward signal is working (print rewards)
- Ensure diversity bonus is active
- Verify opponent model is being used

---

### "Strategy collapse (all same action)"
**Problem:** Agent uses only 1-2 actions

**Solutions:**
- Increase diversity bonus weight
- Add entropy regularization
- Reduce steps per generation (force more resets)
- Train against more diverse opponents

---

### "Training too slow"
**Problem:** Generations take very long

**Solutions:**
- Reduce steps per generation (50k → 20k)
- Reduce number of actions (25 → 15)
- Use smaller state space (32 → 16)
- Reduce population size (5 → 3)

---

### "No clear winner (stuck at 0.5)"
**Problem:** State dominance always ~0.5

**Solutions:**
- This might be OK! Could indicate balanced strategies
- Or increase reward signal strength
- Or add asymmetry to environment

---

## 🔬 Advanced: Customization

### Change Action Space Size
```python
trainer = DynamicCoEvolutionTrainer(
    num_actions=50,  # More actions = more exploration space
    state_dim=32
)
```

### Change State Dimensionality
```python
trainer = DynamicCoEvolutionTrainer(
    num_actions=25,
    state_dim=64  # Richer state representation
)
```

### Adjust Diversity Bonus
```python
# In DynamicCoEvolutionEnv._diversity_bonus()
bonus = diversity_ratio * 10.0  # Increase from 5.0
```

### Modify State Dynamics
```python
# In DynamicCoEvolutionEnv._dynamic_state_transition()
affected_dims = np.random.choice(
    self.state_dim,
    size=self.state_dim // 2,  # Affect more dimensions
    replace=False
)
```

---

## 📚 Key Files

| File | Purpose |
|------|---------|
| `aceac_dynamic_coevolution.py` | Main dynamic training system |
| `DYNAMIC_VS_STATIC_COMPARISON.md` | Detailed comparison of approaches |
| `README_DYNAMIC_SYSTEM.md` | This file - usage guide |

---

## 🎓 Learning Outcomes

After training with dynamic system:

✅ **Agents learn:**
- Which actions are effective (through experience)
- When to use each action (contextual)
- How to counter opponent strategies (adaptive)
- Optimal sequences (emergent)

✅ **You learn:**
- What strategies emerge naturally
- Which actions agents prefer (discovered, not coded)
- How co-evolution drives innovation
- Limitations of pure learning (if any)

❌ **Agents DON'T learn:**
- Predetermined action meanings
- Hardcoded effectiveness values
- Fixed reward formulas
- Human-designed strategies

---

## 🚀 Next Steps

1. **Run baseline training**
   ```bash
   python aceac_dynamic_coevolution.py
   ```

2. **Analyze results**
   - Check training_history.json
   - Visualize state dominance over time
   - Examine action diversity trends

3. **Test trained agents**
   - Load models and run evaluations
   - Observe emergent strategies
   - Compare vs static system

4. **Experiment**
   - Try different action space sizes
   - Adjust state dimensions
   - Modify reward signals
   - Add domain-specific constraints (minimal!)

---

## 🎯 Success Criteria

**Good Training:**
- ✅ Action diversity > 0.3
- ✅ State dominance oscillates (co-evolution)
- ✅ Strategies change over generations
- ✅ No single dominant action

**Poor Training:**
- ❌ All actions same (diversity <0.1)
- ❌ State dominance stuck at 0.5 or 1.0
- ❌ No change after generation 5
- ❌ Single action used 90%+ of time

---

## 💡 Philosophy

**Remember:**

> "The best cybersecurity strategies are the ones we haven't thought of yet. Let the agents discover them."

- No assumptions about "correct" attack sequences
- No predetermined tool effectiveness
- No hardcoded defense strategies
- Pure learning, pure adaptation, pure emergence

**Trust the learning process. Embrace the unknown.**

---

**Train Tomorrow's Defenders - Through Pure Learning, Not Programming!** 🧠🛡️

---

**Version:** 2.0-DYNAMIC
**Last Updated:** 2025-11-18
**Maintained By:** @sarowarzahan414
