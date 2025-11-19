# Dynamic vs Static: Removing All Predefined Logic

**Author:** @sarowarzahan414
**Date:** 2025-11-18
**Philosophy:** Fully adaptive systems learn everything through experience

---

## Overview

This document compares the **static (hardcoded logic)** approach vs the **dynamic (learned behavior)** approach for adversarial RL in cybersecurity.

---

## Key Principle

> **"The system's core functionality must be fully adaptive and dynamic, relying on learned behaviors rather than predefined logic or static commands."**

---

## Static Approach (OLD) ❌

### Hardcoded Attack Powers
```python
# aceac_cyber_range.py - STATIC
def _get_attack_power(self, action):
    powers = {
        0: 0.08,  # Scan - predefined!
        1: 0.12,  # Probe - predefined!
        2: 0.18,  # Exploit - predefined!
        3: 0.22,  # Privilege Escalation - predefined!
        4: 0.15,  # Lateral Movement - predefined!
        5: 0.16,  # Exfiltration - predefined!
        6: 0.14,  # Persistence - predefined!
        7: 0.10,  # C2 - predefined!
        8: 0.20,  # Ransomware - predefined!
        9: 0.09   # Cover Tracks - predefined!
    }
    return float(powers[action])
```

**Problems:**
- ❌ Agents don't learn, they exploit hardcoded values
- ❌ No discovery of novel strategies
- ❌ Predetermined "correct" sequences
- ❌ Limits emergent behavior

---

### Hardcoded Tool-Phase Mappings
```python
# aceac_v2_cyber_killchain.py - STATIC
tool_phase_map = {
    # Reconnaissance tools (0-4) → RECONNAISSANCE phase
    0: CyberKillChainPhase.RECONNAISSANCE,
    1: CyberKillChainPhase.RECONNAISSANCE,
    # ... more predefined mappings

    # Exploitation tools (10-14) → EXPLOITATION phase
    10: CyberKillChainPhase.EXPLOITATION,
    11: CyberKillChainPhase.EXPLOITATION,
    # ... hardcoded semantics
}
```

**Problems:**
- ❌ Tools have predefined meanings
- ❌ Agents can't discover new tool uses
- ❌ Fixed progression paths
- ❌ No creative strategy development

---

### Hardcoded Reward Functions
```python
# aceac_coevolution_FIXED.py - STATIC
if self.agent_role == "red":
    attack_power = 0.08 + action * 0.015  # Linear formula!
    self.network_health -= attack_power
    self.red_score += attack_power * 100
    reward = attack_power * 100  # Predetermined reward!
else:
    defense_power = 0.05 + action * 0.012  # Linear formula!
    self.network_health = min(1.0, self.network_health + defense_power)
    self.blue_score += defense_power * 100
    reward = defense_power * 100  # Predetermined reward!
```

**Problems:**
- ❌ Reward directly tied to predefined formulas
- ❌ Agents optimize formulas, not real objectives
- ❌ No learned value functions
- ❌ Brittle to environment changes

---

## Dynamic Approach (NEW) ✅

### No Predefined Action Effects
```python
# aceac_dynamic_coevolution.py - DYNAMIC
def _dynamic_state_transition(self, action, opponent_action):
    """
    State transition with NO predefined logic

    Actions have RANDOM initial effects
    Success emerges through:
    - What gets rewarded by outcomes
    - What works against opponents
    - Emergent patterns from training
    """
    new_state = self.state.copy()

    # Use action as seed - deterministic but not hardcoded
    np.random.seed(action)
    affected_dims = np.random.choice(
        self.state_dim,
        size=self.state_dim // 4,
        replace=False
    )

    for dim in affected_dims:
        # Effect size varies randomly
        effect_size = np.random.uniform(0.05, 0.15)
        direction = 1.0 if self.agent_role == "red" else -1.0
        new_state[dim] += direction * effect_size

    # NO hardcoded "action X does Y"
    # Effectiveness emerges from what gets rewarded
```

**Benefits:**
- ✅ Actions have no predetermined meanings
- ✅ Agents discover what works through trial and error
- ✅ Novel strategies can emerge
- ✅ Fully adaptive to changing conditions

---

### Minimal Reward Signal
```python
# aceac_dynamic_coevolution.py - DYNAMIC
def _calculate_adaptive_reward(self, action, opponent_action):
    """
    Adaptive reward with MINIMAL hardcoded assumptions

    Based on:
    1. State trajectory (did things improve?)
    2. Diversity bonus (encourage exploration)
    3. Interaction outcomes (who "won"?)

    NO hardcoded effectiveness values!
    """
    # Simple: did state move in favorable direction?
    if self.agent_role == "red":
        state_reward = (current_state_mean - prev_state_mean) * 100.0
    else:
        state_reward = (prev_state_mean - current_state_mean) * 100.0

    # Diversity bonus
    diversity_reward = self._diversity_bonus()

    # Interaction outcome
    state_advantage = np.mean(self.state) - 0.5
    if self.agent_role == "red":
        interaction_reward = state_advantage * 10.0
    else:
        interaction_reward = -state_advantage * 10.0

    return state_reward + diversity_reward + interaction_reward
```

**Benefits:**
- ✅ Reward based on outcomes, not formulas
- ✅ Agents learn value functions from experience
- ✅ Encourages exploration and diversity
- ✅ Adapts to opponent strategies

---

### No Semantic State Features
```python
# STATIC (OLD) - Hardcoded features
obs[0] = self.network_health      # "network health" - semantic!
obs[1] = self.red_score / 100.0   # "red score" - semantic!
obs[2] = self.blue_score / 100.0  # "blue score" - semantic!
obs[3] = self.kill_chain_phase    # "kill chain" - semantic!
# Each feature has predetermined meaning

# DYNAMIC (NEW) - Pure state
obs = self.state  # Just numbers, no semantics!
# Agents learn what matters through training
```

**Benefits:**
- ✅ No predetermined state meanings
- ✅ Agents learn relevant features
- ✅ Representation learning happens naturally
- ✅ Flexible to different scenarios

---

## Comparison Table

| Aspect | Static (OLD) ❌ | Dynamic (NEW) ✅ |
|--------|----------------|------------------|
| **Action Meanings** | Predefined (Scan, Exploit, etc.) | Learned through experience |
| **Action Effects** | Hardcoded power values | Emergent from training |
| **Tool-Phase Map** | Fixed mappings | No predetermined sequence |
| **Reward Function** | Complex formulas | Minimal sparse signal |
| **State Features** | Semantic (health, score) | Pure numeric state |
| **Strategy Development** | Exploits hardcoded values | Discovers through self-play |
| **Adaptability** | Limited to coded scenarios | Fully adaptive |
| **Novel Behaviors** | Impossible | Encouraged |
| **Opponent Modeling** | Uses known weaknesses | Learns weaknesses |
| **Realism** | Depends on coding accuracy | Emerges from dynamics |

---

## Examples: What Changes

### Example 1: Attack Selection

**Static Approach:**
```python
# Agent learns: "Action 2 (Exploit) is worth 0.18"
# Agent learns: "Action 8 (Ransomware) is worth 0.20"
# Strategy: Always use action 8 (highest hardcoded value)
```

**Dynamic Approach:**
```python
# Agent tries: Action 2 → State improves by 0.3 → +30 reward
# Agent tries: Action 8 → State improves by 0.1 → +10 reward
# Agent tries: Action 15 → State improves by 0.5 → +50 reward
# Strategy: Use action 15 (learned effectiveness)
# NOTE: Action 15 might be completely different next run!
```

---

### Example 2: Defense Against Attack

**Static Approach:**
```python
# Blue knows: Red uses action 12 (SQL injection)
# Blue knows: Action 16 (Fail2ban) blocks SQL injection
# Strategy: Hardcoded counter based on tool semantics
```

**Dynamic Approach:**
```python
# Blue observes: When Red uses action 12, state moves toward Red
# Blue tries: Action 5 → State doesn't change much
# Blue tries: Action 16 → State moves back toward Blue
# Strategy: Learn action 16 counters action 12 through experience
# Discovers this relationship, doesn't use predefined knowledge
```

---

### Example 3: Strategy Evolution

**Static Approach:**
```python
# Generation 1: Red uses highest power attacks (hardcoded)
# Generation 10: Red still uses highest power attacks
# No evolution, just exploiting hardcoded values
```

**Dynamic Approach:**
```python
# Generation 1: Red tries random actions, learns what works
# Generation 5: Red discovers action sequence [3, 7, 15] works well
# Generation 10: Blue learns to counter [3, 7, 15]
# Generation 15: Red discovers new sequence [22, 1, 9]
# Continuous co-evolution of novel strategies!
```

---

## Training Process Comparison

### Static Training
```
1. Environment has hardcoded rules
2. Agent learns to exploit those rules
3. Optimal strategy = exploit hardcoded values
4. Limited by quality of hardcoding
5. No true emergence
```

### Dynamic Training
```
1. Environment has minimal assumptions
2. Agent explores randomly initially
3. Gradient ascent on what works
4. Opponent adapts, forcing innovation
5. True emergent strategies
6. Continuous co-evolution
```

---

## Key Philosophy

### Static = Engineering

You **design** the system:
- Define what each action does
- Specify effectiveness values
- Create reward formulas
- Set progression paths

**Result:** Agents learn to navigate your design, not solve the problem.

---

### Dynamic = Science

You **observe** emergent behavior:
- Actions have random initial effects
- Effectiveness emerges from experience
- Rewards based on outcomes
- Strategies develop naturally

**Result:** Agents learn to solve the problem, discovering novel solutions.

---

## Migration Path

### Step 1: Remove Hardcoded Values ✅
```python
# Before
attack_power = 0.18  # Hardcoded

# After
attack_power = learned_from_experience()
```

### Step 2: Simplify Reward ✅
```python
# Before
reward = attack_power * 100 + stealth_bonus - detection_penalty

# After
reward = state_improved() ? +1 : -1  # Sparse
```

### Step 3: Remove Semantic Features ✅
```python
# Before
obs = [network_health, red_score, blue_score, kill_chain_phase]

# After
obs = raw_state  # Pure numbers
```

### Step 4: Trust the Learning ✅
```python
# Before
if phase < EXPLOITATION:
    block_action()  # Enforce rules

# After
# No enforcement, agents learn valid sequences
```

---

## Expected Outcomes

### With Dynamic System:

✅ **Emergent Strategies**
- Agents discover attack patterns not hardcoded
- Novel defense mechanisms emerge
- Unexpected but effective behaviors

✅ **True Co-Evolution**
- Red adapts to Blue's learned defenses
- Blue adapts to Red's learned attacks
- Continuous arms race

✅ **Generalization**
- Agents learn principles, not rules
- Transfer to new scenarios
- Robust to changes

✅ **Discovery**
- Find optimal strategies through exploration
- May discover better approaches than human design
- Innovation through learning

❌ **Potential Challenges**
- Longer training time (no shortcuts)
- May need more samples
- Less interpretable initially
- Requires careful monitoring for edge cases

---

## Implementation Guide

### Quick Start: Dynamic Training

```bash
# Run fully dynamic co-evolution
python aceac_dynamic_coevolution.py

# What you'll see:
# - Agents start random
# - Gradually learn effective actions
# - Strategies evolve over generations
# - No hardcoded shortcuts
```

### Monitoring Dynamic Training

```python
from aceac_dynamic_coevolution import DynamicCoEvolutionTrainer

trainer = DynamicCoEvolutionTrainer(
    num_actions=25,    # No predefined meanings!
    state_dim=32,      # Pure state space
    population_size=5
)

# Train and watch emergence
red, blue = trainer.train(
    num_generations=20,
    steps_per_generation=50000
)

# Evaluate learned behaviors
# Actions will have meanings based on what was rewarded
# Not based on what was hardcoded
```

---

## Conclusion

### Static Systems (OLD):
- Engineer behavior through hardcoded rules
- Agents exploit predefined logic
- Limited to designer's knowledge
- **Not truly adaptive**

### Dynamic Systems (NEW):
- Behavior emerges through learning
- Agents discover effective strategies
- Unlimited by design constraints
- **Fully adaptive and innovative**

---

## Philosophy Statement

**"Don't tell the agents what to do. Let them learn what works."**

- No predetermined attack powers
- No hardcoded tool meanings
- No fixed reward formulas
- No semantic state features

**Pure learning. Pure adaptation. Pure emergence.**

---

**The best strategies are the ones we didn't code.**

Train Tomorrow's Defenders - Through Pure Learning! 🧠
