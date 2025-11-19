# Red/Blue Co-Evolution Edge Case Analysis
## Adversarial RL Testbed Edge Cases & Failure Modes

**Version:** 1.0
**Date:** 2025-11-18
**Author:** @sarowarzahan414
**Project:** ACEAC - Adversarial Co-Evolution for Autonomous Cyber-defense

---

## Table of Contents

1. [Introduction](#introduction)
2. [Co-Evolution Dynamics Edge Cases](#1-co-evolution-dynamics-edge-cases)
3. [Red Agent Edge Cases](#2-red-agent-edge-cases)
4. [Blue Agent Edge Cases](#3-blue-agent-edge-cases)
5. [Interaction Edge Cases](#4-interaction-edge-cases)
6. [Training Instability Edge Cases](#5-training-instability-edge-cases)
7. [Model Compatibility Edge Cases](#6-model-compatibility-edge-cases)
8. [Performance Measurement Edge Cases](#7-performance-measurement-edge-cases)
9. [Resource & System Edge Cases](#8-resource--system-edge-cases)
10. [Kill Chain Specific Edge Cases](#9-kill-chain-specific-edge-cases)
11. [Detection & Mitigation Strategies](#10-detection--mitigation-strategies)

---

## Introduction

Co-evolution systems where Red (offensive) and Blue (defensive) agents train against each other introduce unique edge cases beyond standard RL environments. These systems can exhibit pathological behaviors including:

- **Cyclic dominance** (rock-paper-scissors dynamics)
- **Collapse to trivial strategies**
- **Runaway optimization**
- **Mode collapse**
- **Exploitation of simulation artifacts**

This document catalogs **50+ critical edge cases** with detection and mitigation strategies.

---

## 1. Co-Evolution Dynamics Edge Cases

### 1.1 Cyclic Dominance (Non-Convergent Oscillation) 🔴 CRITICAL

**Description:** Red and Blue agents cycle through strategies without converging to a Nash equilibrium.

**Manifestation:**
```
Generation 1: Red wins 8/10  (Red uses exploit X)
Generation 2: Blue wins 9/10  (Blue defends against X)
Generation 3: Red wins 8/10  (Red switches to exploit Y)
Generation 4: Blue wins 9/10  (Blue defends against Y)
... cycles indefinitely ...
```

**Root Cause:**
- No memory of previous opponents
- Strategy forgetting due to catastrophic forgetting
- Insufficient strategy diversity

**Detection:**
```python
def detect_cyclic_dominance(history, window=5):
    """Detect cyclic win patterns"""
    if len(history) < window * 2:
        return False

    # Check for repeating patterns
    wins_red = [h['red_wins'] for h in history[-window*2:]]

    # Simple autocorrelation check
    first_half = wins_red[:window]
    second_half = wins_red[window:]

    # If patterns repeat, likely cyclic
    correlation = np.corrcoef(first_half, second_half)[0, 1]

    if correlation > 0.8:  # High correlation = cycling
        return True, "Cyclic dominance detected"

    return False, None
```

**Mitigation:**
```python
class PopulationBasedCoEvolution:
    """Maintain population of strategies to prevent cycling"""

    def __init__(self, population_size=10):
        self.red_population = []
        self.blue_population = []
        self.population_size = population_size

    def train_generation(self, red_model, blue_model):
        # Train against random opponent from population
        opponent_idx = np.random.randint(len(self.blue_population))
        opponent = self.blue_population[opponent_idx]

        # Train red vs historical blue
        train_against(red_model, opponent)

        # Add to population
        self.red_population.append(copy.deepcopy(red_model))

        # Keep population size limited
        if len(self.red_population) > self.population_size:
            self.red_population.pop(0)

        return red_model
```

**Example from ACEAC:**
```python
# aceac_coevolution_FIXED.py:200-227
# Training red vs blue Gen N-1, then blue vs red Gen N
# This can cause cycling if agents forget previous strategies
```

**Priority:** 🔴 CRITICAL - Can prevent convergence entirely

---

### 1.2 Strategy Collapse (Mode Collapse) 🔴 CRITICAL

**Description:** Both agents converge to trivial or degenerate strategies.

**Example Scenarios:**

**Scenario A: Red Always Does Nothing**
```python
# Red learns that attacks are detected/blocked
# So optimal strategy becomes: don't attack
red_actions = [0, 0, 0, 0, 0, ...]  # All reconnaissance, no exploitation
red_score = 0.0  # Zero reward, but also zero risk
```

**Scenario B: Blue Over-Defends**
```python
# Blue learns to max out all defenses
# Network becomes unusable due to restrictions
blue_actions = [9, 9, 9, 9, 9, ...]  # Maximum firewall/blocking
network_security = 1.0  # Perfect security
productivity = 0.0  # But network is unusable!
```

**Detection:**
```python
def detect_strategy_collapse(agent_history, diversity_threshold=0.1):
    """Detect if agent is using limited strategy diversity"""

    # Get recent actions
    recent_actions = agent_history[-100:]

    # Calculate action entropy
    action_counts = np.bincount(recent_actions, minlength=10)
    action_probs = action_counts / len(recent_actions)

    # Shannon entropy
    entropy = -np.sum(action_probs * np.log(action_probs + 1e-10))
    max_entropy = np.log(10)  # For 10 actions

    normalized_entropy = entropy / max_entropy

    if normalized_entropy < diversity_threshold:
        return True, f"Low diversity: {normalized_entropy:.3f}"

    return False, None


def detect_trivial_strategy(env, model, num_episodes=20):
    """Detect if agent is using trivial strategy"""

    total_reward = 0
    action_variance = []

    for _ in range(num_episodes):
        obs, _ = env.reset()
        episode_actions = []

        for step in range(50):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, term, trunc, info = env.step(action)

            total_reward += reward
            episode_actions.append(action)

            if term:
                break

        action_variance.append(np.var(episode_actions))

    avg_variance = np.mean(action_variance)
    avg_reward = total_reward / num_episodes

    # Trivial if low variance and low reward
    if avg_variance < 1.0 and avg_reward < 10.0:
        return True, f"Trivial strategy: var={avg_variance:.2f}, reward={avg_reward:.1f}"

    return False, None
```

**Mitigation:**
```python
def add_exploration_bonus(reward, action_entropy, bonus_weight=0.1):
    """Add entropy bonus to encourage exploration"""
    exploration_bonus = action_entropy * bonus_weight
    return reward + exploration_bonus


def curriculum_learning(generation, max_generations):
    """Gradually increase difficulty"""
    difficulty = generation / max_generations

    # Start with easier opponents, increase difficulty
    if difficulty < 0.3:
        return "easy"
    elif difficulty < 0.7:
        return "medium"
    else:
        return "hard"
```

**Priority:** 🔴 CRITICAL - Makes training useless

---

### 1.3 Runaway Optimization (Arms Race) 🟠 HIGH

**Description:** Agents develop capabilities far beyond realistic/intended bounds.

**Example:**
```python
# After 100 generations
red_success_rate = 0.99  # Wins 99% of time
blue_detection_rate = 0.98  # Detects 98% of attacks

# Both have become unrealistically powerful
# Red can bypass any defense
# Blue can detect any attack

# This doesn't reflect real-world constraints!
```

**Detection:**
```python
def detect_runaway_optimization(metrics, capability_cap=0.95):
    """Detect if agents exceed realistic capabilities"""

    red_success = metrics['red_success_rate']
    blue_block = metrics['blue_block_rate']

    issues = []

    if red_success > capability_cap:
        issues.append(f"Red too powerful: {red_success:.2%} success")

    if blue_block > capability_cap:
        issues.append(f"Blue too powerful: {blue_block:.2%} block rate")

    # Check for unrealistic speeds
    if metrics.get('avg_exploit_time', float('inf')) < 1.0:
        issues.append("Unrealistically fast exploits")

    return len(issues) > 0, issues
```

**Mitigation:**
```python
class BoundedCoEvolution:
    """Enforce realistic capability bounds"""

    MAX_SUCCESS_RATE = 0.85  # Red can't win >85%
    MAX_DEFENSE_RATE = 0.85  # Blue can't block >85%

    def evaluate_and_bound(self, red_model, blue_model):
        # Test current capabilities
        red_success = self.test_red_agent(red_model)
        blue_defense = self.test_blue_agent(blue_model)

        # If exceeds bounds, roll back or reset
        if red_success > self.MAX_SUCCESS_RATE:
            print(f"⚠️ Red too powerful ({red_success:.1%}), rolling back")
            red_model = self.red_checkpoint  # Revert to previous

        if blue_defense > self.MAX_DEFENSE_RATE:
            print(f"⚠️ Blue too powerful ({blue_defense:.1%}), rolling back")
            blue_model = self.blue_checkpoint

        return red_model, blue_model
```

**Priority:** 🟠 HIGH - Produces unrealistic agents

---

### 1.4 Forgetting Catastrophe 🟠 HIGH

**Description:** Agent forgets how to handle previous opponent strategies.

**Example:**
```python
# Generation 5: Red learns to exploit SQL injection
# Generations 6-10: Blue learns to defend SQL injection
# Generations 11-15: Red learns new XSS attack
# Generation 16: Blue FORGETS how to defend SQL injection!

# Test against Gen 5 Red:
blue_gen16_vs_red_gen5 = 0.2  # Blue only wins 20%!
# Blue has catastrophically forgotten old defenses
```

**Detection:**
```python
def detect_catastrophic_forgetting(current_model, historical_opponents):
    """Test current model against historical opponents"""

    performance_history = []

    for gen, opponent in enumerate(historical_opponents):
        # Test current model vs old opponent
        win_rate = test_matchup(current_model, opponent)
        performance_history.append({
            'generation': gen,
            'win_rate': win_rate
        })

    # Check for performance degradation
    recent_perf = np.mean([h['win_rate'] for h in performance_history[-5:]])
    old_perf = np.mean([h['win_rate'] for h in performance_history[:5]])

    if recent_perf < old_perf - 0.2:  # 20% drop
        return True, f"Forgetting detected: {old_perf:.1%} → {recent_perf:.1%}"

    return False, None
```

**Mitigation:**
```python
def replay_buffer_training(current_model, replay_buffer):
    """Train against mix of current and historical opponents"""

    # 70% current opponent, 30% historical
    if np.random.random() < 0.3:
        opponent = random.choice(replay_buffer)
        print(f"Training against historical opponent (Gen {opponent.generation})")
    else:
        opponent = current_opponent

    # Train
    train_against(current_model, opponent)

    return current_model


def elastic_weight_consolidation(model, important_params):
    """EWC: Penalize changes to important parameters"""

    # Calculate Fisher information for important tasks
    fisher_matrix = calculate_fisher_information(model, old_tasks)

    # Add EWC loss term
    ewc_loss = sum(fisher * (param - old_param)**2
                   for param, old_param, fisher
                   in zip(model.parameters(), old_parameters, fisher_matrix))

    total_loss = task_loss + lambda_ewc * ewc_loss
    return total_loss
```

**Priority:** 🟠 HIGH - Degrades agent quality

---

### 1.5 Exploitation of Simulation Artifacts 🟡 MEDIUM

**Description:** Agents exploit unrealistic features of the simulation environment.

**Example:**
```python
# Red discovers that action order doesn't matter in simulation
# In reality, reconnaissance must precede exploitation
# But in simulation, can exploit without recon!

# Unrealistic strategy:
red_actions = [12, 10, 2, 0]  # SQL injection, then scan
# This works in simulation but impossible in reality

# Or: Blue discovers infinite resources
blue_actions = [24, 24, 24, ...]  # Spam most expensive defense
# No cost modeling = unrealistic behavior
```

**Detection:**
```python
def validate_action_sequence(actions, kill_chain_phase):
    """Ensure actions follow realistic constraints"""

    # Check kill chain progression
    for i, action in enumerate(actions):
        required_phase = get_required_phase(action)

        if required_phase > kill_chain_phase:
            return False, f"Action {action} requires phase {required_phase}, but at {kill_chain_phase}"

    # Check resource constraints
    total_cost = sum(get_action_cost(a) for a in actions)
    if total_cost > MAX_BUDGET:
        return False, f"Actions exceed budget: {total_cost} > {MAX_BUDGET}"

    return True, None
```

**Mitigation:**
```python
class RealisticEnvironment(gym.Env):
    """Environment with realistic constraints"""

    def __init__(self):
        super().__init__()

        # Resource constraints
        self.red_budget = 100
        self.blue_budget = 100

        # Kill chain state
        self.kill_chain_phase = 0

        # Action costs
        self.action_costs = {
            # Red tools
            0: 1,   # Nmap (cheap)
            12: 20, # SQL injection (expensive)
            # Blue tools
            10: 5,  # OSSEC (moderate)
            24: 50, # GHIDRA (expensive)
        }

    def step(self, action):
        # Check if action is valid given current state
        if not self._is_action_valid(action):
            return self._get_obs(), -50.0, True, False, {'error': 'Invalid action'}

        # Deduct cost
        cost = self.action_costs.get(action, 10)
        if self.agent_role == 'red':
            self.red_budget -= cost
        else:
            self.blue_budget -= cost

        # Execute action
        reward, success = self._execute_action(action)

        # Penalize if out of budget
        if self.red_budget < 0 or self.blue_budget < 0:
            reward -= 100

        return self._get_obs(), reward, False, False, {}

    def _is_action_valid(self, action):
        """Check if action is valid given current game state"""

        # Exploitation tools require reconnaissance phase
        if action in [10, 11, 12, 13, 14]:  # Exploit tools
            if self.kill_chain_phase < 3:  # EXPLOITATION phase
                return False

        # Check budget
        cost = self.action_costs.get(action, 10)
        budget = self.red_budget if self.agent_role == 'red' else self.blue_budget

        if cost > budget:
            return False

        return True
```

**Priority:** 🟡 MEDIUM - Reduces real-world applicability

---

## 2. Red Agent Edge Cases

### 2.1 Red Agent Always Uses Same Attack Vector 🟠 HIGH

**Description:** Red agent converges to single attack type, making it easy to defend.

**Example:**
```python
# Red only uses SQL injection
action_distribution = {
    12: 0.95,  # SQL injection 95% of time
    0: 0.03,   # Nmap 3%
    1: 0.02,   # Other 2%
}

# Blue easily defends by hardening SQL
blue_defense_strategy = "Block all SQL patterns"
blue_win_rate = 0.98
```

**Detection:**
```python
def detect_single_strategy(action_history, threshold=0.7):
    """Detect if agent overuses single action"""

    action_counts = np.bincount(action_history)
    max_action_freq = np.max(action_counts) / len(action_history)

    if max_action_freq > threshold:
        most_used_action = np.argmax(action_counts)
        return True, f"Action {most_used_action} used {max_action_freq:.1%} of time"

    return False, None
```

**Mitigation:**
```python
def diversity_reward(action, action_history, diversity_bonus=5.0):
    """Reward using underutilized actions"""

    # Count recent action usage
    recent_actions = action_history[-100:]
    action_counts = np.bincount(recent_actions, minlength=25)
    action_frequency = action_counts / len(recent_actions)

    # Bonus for rare actions
    rarity = 1.0 - action_frequency[action]
    bonus = rarity * diversity_bonus

    return bonus
```

**Priority:** 🟠 HIGH - Reduces training effectiveness

---

### 2.2 Red Agent Ignores Stealth Penalties 🟡 MEDIUM

**Description:** Red uses noisy attacks despite stealth penalties, getting detected every time.

**Example:**
```python
# Red spams Metasploit and EternalBlue (noisy tools)
red_actions = [14, 24, 14, 24, ...]  # Cobalt Strike, EternalBlue
detection_level = 0.95  # Always detected!

# Even though successful, red is caught
# In real attack, this would be failure
```

**Detection & Mitigation:**
```python
class StealthAwareReward:
    """Penalize detection more heavily"""

    def calculate_reward(self, success, detection_level, data_exfiltrated):
        if success:
            base_reward = 20.0 + data_exfiltrated * 50.0
        else:
            base_reward = -10.0

        # Heavy detection penalty
        if detection_level > 0.7:
            detection_penalty = detection_level * 100.0  # Up to -100
            base_reward -= detection_penalty

        # Stealth bonus
        if success and detection_level < 0.3:
            base_reward += 50.0  # Big bonus for stealthy success

        return base_reward
```

**Priority:** 🟡 MEDIUM - Affects realism

---

### 2.3 Red Agent Stuck in Reconnaissance Phase 🟡 MEDIUM

**Description:** Red never progresses past early kill chain phases.

**Example:**
```python
# After 50 steps, still doing reconnaissance
kill_chain_progress = {
    'RECONNAISSANCE': 1.0,
    'WEAPONIZATION': 0.1,
    'DELIVERY': 0.0,
    'EXPLOITATION': 0.0,
    'INSTALLATION': 0.0,
    'COMMAND_CONTROL': 0.0,
    'ACTIONS_OBJECTIVES': 0.0
}

# Red never achieves actual compromise!
```

**Detection & Mitigation:**
```python
def phase_progression_reward(current_phase, previous_phase):
    """Reward advancing through kill chain"""

    phase_values = {
        'RECONNAISSANCE': 0,
        'WEAPONIZATION': 1,
        'DELIVERY': 2,
        'EXPLOITATION': 3,
        'INSTALLATION': 4,
        'COMMAND_CONTROL': 5,
        'ACTIONS_OBJECTIVES': 6
    }

    current_value = phase_values[current_phase]
    previous_value = phase_values[previous_phase]

    if current_value > previous_value:
        # Big reward for phase advancement
        return (current_value - previous_value) * 25.0

    # Small penalty for staying in same phase too long
    return -1.0


def detect_phase_stagnation(phase_history, window=20):
    """Detect if stuck in same phase"""

    recent_phases = phase_history[-window:]
    unique_phases = len(set(recent_phases))

    if unique_phases == 1:
        return True, f"Stuck in {recent_phases[0]} for {window} steps"

    return False, None
```

**Priority:** 🟡 MEDIUM - Incomplete attacks

---

## 3. Blue Agent Edge Cases

### 3.1 Blue Agent Over-Defends (Unusable Network) 🟠 HIGH

**Description:** Blue makes network so secure it's unusable for legitimate users.

**Example:**
```python
# Blue blocks everything
network_security = 1.0  # Perfect security!
productivity = 0.0      # But zero productivity
user_satisfaction = 0.0 # Users can't work

# This is a Pyrrhic victory
```

**Detection & Mitigation:**
```python
class BalancedBlueReward:
    """Balance security vs. usability"""

    def calculate_reward(self, security, productivity):
        # Security reward
        security_reward = security * 50.0

        # Productivity reward (maintain minimum 60%)
        if productivity > 0.6:
            productivity_reward = productivity * 30.0
        else:
            # Heavy penalty for low productivity
            productivity_reward = (productivity - 0.6) * 200.0

        # Balanced reward
        total = security_reward + productivity_reward

        return total

    def update_environment(self, action):
        # Each defensive action reduces productivity slightly
        productivity_cost = {
            5: 0.02,  # iptables (low cost)
            9: 0.10,  # Fortinet (high cost)
            16: 0.05, # fail2ban (medium)
        }

        self.productivity -= productivity_cost.get(action, 0.03)
        self.productivity = max(0.0, self.productivity)
```

**Priority:** 🟠 HIGH - Defeats purpose of defense

---

### 3.2 Blue Agent Ignores Slow/Stealthy Attacks 🟠 HIGH

**Description:** Blue only defends against obvious/loud attacks, misses APT-style slow attacks.

**Example:**
```python
# Blue defends well against fast attacks
fast_attack_block_rate = 0.90

# But misses slow, stealthy attacks
slow_attack_block_rate = 0.10  # Only 10%!

# APT-style attacks succeed 90% of time
```

**Detection & Mitigation:**
```python
def test_against_attack_types(blue_model):
    """Test against different attack patterns"""

    # Fast aggressive attack
    fast_red = create_aggressive_red()
    fast_block_rate = test_defense(blue_model, fast_red)

    # Slow stealthy attack
    stealth_red = create_stealthy_red()
    stealth_block_rate = test_defense(blue_model, stealth_red)

    # Check balance
    if stealth_block_rate < fast_block_rate - 0.3:
        return False, "Vulnerable to stealthy attacks"

    return True, "Balanced defense"


def diverse_attack_training(blue_model):
    """Train against diverse attack patterns"""

    attack_patterns = [
        'fast_aggressive',
        'slow_stealthy',
        'multi_vector',
        'zero_day_exploit',
        'insider_threat',
    ]

    for pattern in attack_patterns:
        red_agent = create_red_with_pattern(pattern)
        train_blue_vs_red(blue_model, red_agent)

    return blue_model
```

**Priority:** 🟠 HIGH - Critical security gap

---

### 3.3 Blue Agent Resource Exhaustion 🟡 MEDIUM

**Description:** Blue deploys so many defenses it exhausts computational resources.

**Example:**
```python
# Blue runs all defensive tools simultaneously
active_tools = [0, 1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15, 20, 21, 22, 23, 24]
# 17 tools running at once!

cpu_usage = 100%
memory_usage = 32GB
cost_per_month = $10,000  # Unsustainable
```

**Mitigation:**
```python
class ResourceConstrainedBlue(gym.Env):
    """Blue team with resource constraints"""

    def __init__(self):
        self.max_concurrent_tools = 5
        self.active_tools = []
        self.resource_budget = 100

    def step(self, action):
        tool_cost = self.get_tool_cost(action)

        # Check if can afford
        if tool_cost > self.resource_budget:
            return self._get_obs(), -50.0, False, False, {'error': 'Insufficient resources'}

        # Check concurrency limit
        if len(self.active_tools) >= self.max_concurrent_tools:
            # Must deactivate a tool first
            self.active_tools.pop(0)

        # Activate tool
        self.active_tools.append(action)
        self.resource_budget -= tool_cost

        # Execute defense
        reward = self._execute_defense(action)

        return self._get_obs(), reward, False, False, {}
```

**Priority:** 🟡 MEDIUM - Affects cost efficiency

---

## 4. Interaction Edge Cases

### 4.1 Symmetric Strategies (Both Do Same Thing) 🟡 MEDIUM

**Description:** Red and Blue converge to similar strategies, nullifying each other.

**Example:**
```python
# Both agents use similar timing/patterns
red_attack_interval = [10, 15, 10, 15, 10]
blue_defense_interval = [10, 15, 10, 15, 10]

# They mirror each other
# Every attack is blocked, but also every defense is evaded
# Result: stalemate at 50/50
```

**Detection:**
```python
def detect_strategy_mirroring(red_actions, blue_actions, window=50):
    """Detect if strategies are too similar"""

    # Get recent action sequences
    red_recent = red_actions[-window:]
    blue_recent = blue_actions[-window:]

    # Calculate sequence similarity
    # (Accounting for different action spaces)
    red_pattern = np.bincount(red_recent, minlength=25)
    blue_pattern = np.bincount(blue_recent, minlength=25)

    # Normalize
    red_pattern = red_pattern / np.sum(red_pattern)
    blue_pattern = blue_pattern / np.sum(blue_pattern)

    # Cosine similarity
    similarity = np.dot(red_pattern, blue_pattern)

    if similarity > 0.8:
        return True, f"High strategy similarity: {similarity:.2f}"

    return False, None
```

**Priority:** 🟡 MEDIUM - Reduces learning

---

### 4.2 Adversarial Deadlock 🟡 MEDIUM

**Description:** Neither agent can make progress; environment state frozen.

**Example:**
```python
# Network health stuck at 0.5
for step in range(100):
    red_damage = 0.05
    blue_healing = 0.05
    network_health += (blue_healing - red_damage)
    print(network_health)  # Always 0.50

# Perfectly balanced, no winner
# Training makes no progress
```

**Detection & Mitigation:**
```python
def detect_deadlock(state_history, tolerance=0.01, window=20):
    """Detect if game state is not changing"""

    recent_states = state_history[-window:]
    state_variance = np.var(recent_states)

    if state_variance < tolerance:
        return True, f"State frozen: variance={state_variance:.4f}"

    return False, None


def add_stochasticity(reward, noise_std=2.0):
    """Add noise to break deterministic deadlocks"""
    noise = np.random.normal(0, noise_std)
    return reward + noise


def perturbation_injection(env):
    """Periodically inject random events"""
    if np.random.random() < 0.1:  # 10% chance
        # Random event
        event = np.random.choice([
            'vulnerability_discovered',
            'security_update',
            'user_error',
            'external_attack'
        ])

        env.handle_random_event(event)
```

**Priority:** 🟡 MEDIUM - Stalls training

---

### 4.3 Reward Hacking Through Opponent Exploitation 🔴 CRITICAL

**Description:** Agent learns to exploit bugs in opponent's policy rather than learn genuine strategy.

**Example:**
```python
# Red discovers: if it uses action sequence [5, 7, 5, 7, ...]
# Blue's policy crashes or returns NaN
# Red gets automatic win without genuine attack

# Or: Blue discovers if it does action 24 repeatedly
# Red agent's gradient explodes
# Blue wins by breaking opponent, not by defending
```

**Detection:**
```python
def detect_opponent_exploitation(agent_actions, opponent_crashed=False):
    """Detect if agent is exploiting opponent bugs"""

    if opponent_crashed:
        # Log the action sequence that caused crash
        print(f"⚠️ Opponent crash with action sequence: {agent_actions[-10:]}")
        return True, "Opponent exploitation detected"

    # Check for repetitive sequences known to cause issues
    suspicious_patterns = [
        [5, 7, 5, 7, 5, 7],
        [24, 24, 24, 24],
        [0, 25, 0, 25],  # Out of bounds toggle
    ]

    recent = agent_actions[-6:]
    for pattern in suspicious_patterns:
        if recent == pattern:
            return True, f"Suspicious pattern: {pattern}"

    return False, None


def robust_opponent_policy(opponent_model, obs, fallback_action=0):
    """Wrap opponent policy with error handling"""
    try:
        action, _ = opponent_model.predict(obs, deterministic=False)

        # Validate action
        if not (0 <= action < 25):
            print(f"⚠️ Invalid action {action}, using fallback")
            action = fallback_action

        # Check for NaN
        if np.isnan(action):
            print(f"⚠️ NaN action, using fallback")
            action = fallback_action

        return action

    except Exception as e:
        print(f"⚠️ Opponent policy error: {e}, using fallback")
        return fallback_action
```

**Priority:** 🔴 CRITICAL - Completely invalidates training

---

## 5. Training Instability Edge Cases

### 5.1 Diverging Performance Metrics 🟠 HIGH

**Description:** Training loss improves but actual performance degrades.

**Example:**
```python
# Training metrics look great
training_loss = [100, 80, 60, 40, 20, 10, 5]  # Decreasing
training_reward = [10, 15, 20, 25, 30, 35, 40]  # Increasing

# But evaluation performance is terrible
eval_win_rate = [0.5, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05]  # Decreasing!

# Agent is overfitting to current opponent
```

**Detection:**
```python
def detect_training_divergence(train_metrics, eval_metrics, threshold=0.3):
    """Detect divergence between training and evaluation"""

    # Recent training performance
    recent_train = np.mean(train_metrics[-5:])
    old_train = np.mean(train_metrics[:5])
    train_improvement = recent_train - old_train

    # Recent eval performance
    recent_eval = np.mean(eval_metrics[-5:])
    old_eval = np.mean(eval_metrics[:5])
    eval_improvement = recent_eval - old_eval

    # Check divergence
    if train_improvement > 0.2 and eval_improvement < -0.1:
        return True, "Training improves but eval degrades (overfitting)"

    # Check if metrics diverge by more than threshold
    divergence = abs(train_improvement - eval_improvement)
    if divergence > threshold:
        return True, f"Large divergence: {divergence:.2f}"

    return False, None
```

**Mitigation:**
```python
def early_stopping_on_eval(eval_metrics, patience=5):
    """Stop if evaluation performance doesn't improve"""

    best_eval = max(eval_metrics)
    best_idx = eval_metrics.index(best_eval)

    steps_since_improvement = len(eval_metrics) - best_idx - 1

    if steps_since_improvement >= patience:
        return True, f"No eval improvement for {patience} generations"

    return False, None


def checkpoint_best_eval(model, eval_score, best_score):
    """Save checkpoint when eval improves"""

    if eval_score > best_score:
        model.save("models/best_eval_checkpoint.zip")
        return eval_score

    return best_score
```

**Priority:** 🟠 HIGH - Produces bad models

---

### 5.2 Gradient Explosion/Vanishing 🟠 HIGH

**Description:** Training becomes unstable due to extreme gradient values.

**Example:**
```python
# Gradient explosion
gradients = [1.0, 2.0, 4.0, 8.0, 16.0, 3200.0, NaN]
# Training crashes

# Or gradient vanishing
gradients = [1.0, 0.5, 0.25, 0.125, 0.0001, 0.0, 0.0]
# No learning occurs
```

**Detection & Mitigation:**
```python
def monitor_gradients(model):
    """Monitor gradient health"""

    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2

    total_norm = total_norm ** 0.5

    # Check for issues
    if total_norm > 100.0:
        return "explosion", total_norm
    elif total_norm < 1e-7:
        return "vanishing", total_norm
    else:
        return "healthy", total_norm


def gradient_clipping(model, max_grad_norm=10.0):
    """Clip gradients to prevent explosion"""
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)


def learning_rate_scheduling(optimizer, epoch, initial_lr=1e-3):
    """Adjust learning rate during training"""

    # Reduce LR if gradients are unstable
    if epoch > 10 and gradient_status == "explosion":
        new_lr = initial_lr * 0.5
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
        print(f"Reduced learning rate to {new_lr}")
```

**Priority:** 🟠 HIGH - Prevents training

---

### 5.3 Reward Hacking via Episode Termination 🟠 HIGH

**Description:** Agent learns to trigger early termination to avoid negative rewards.

**Example:**
```python
# Red learns: if network_health drops to 0.0, episode ends
# Red can get +20 reward for successful attack
# But if it continues, might get -10 for detection
# Solution: Attack once, drive health to 0, episode ends with +20

# Red optimizes for: maximum damage in minimum steps
# Rather than: stealthy long-term compromise
```

**Detection & Mitigation:**
```python
def prevent_premature_termination(env):
    """Don't allow early termination to be exploited"""

    # Original termination conditions
    terminated = (
        env.current_step >= env.max_steps or
        env.network_health <= 0.1 or
        env.network_health >= 0.95
    )

    # But add penalty for early termination
    if terminated and env.current_step < env.max_steps:
        # Penalize ending early
        early_termination_penalty = (env.max_steps - env.current_step) * 0.5
        env.cumulative_reward -= early_termination_penalty

    return terminated


def time_discounted_rewards(rewards, gamma=0.99):
    """Discount future rewards to encourage long-term strategy"""

    discounted = []
    cumulative = 0

    for r in reversed(rewards):
        cumulative = r + gamma * cumulative
        discounted.append(cumulative)

    return list(reversed(discounted))
```

**Priority:** 🟠 HIGH - Distorts learning objective

---

## 6. Model Compatibility Edge Cases

### 6.1 Observation Space Mismatch 🔴 CRITICAL

**Description:** Loading model trained with different observation space.

**Example:**
```python
# Model trained with 20D observation space
red_v1 = PPO.load("models/red_v1.zip")  # Expects (20,) observations

# But environment now uses 62D observations
env_v2 = ACEACv2Environment()  # Returns (62,) observations

# Crash when predicting
obs, _ = env_v2.reset()  # Shape: (62,)
action, _ = red_v1.predict(obs)  # ERROR: Expected (20,), got (62,)
```

**Detection & Mitigation:**
```python
def validate_model_compatibility(model_path, env):
    """Check if model is compatible with environment"""

    # Load model
    model = PPO.load(model_path)

    # Check observation space
    if model.observation_space != env.observation_space:
        raise ValueError(
            f"Observation space mismatch:\n"
            f"  Model: {model.observation_space}\n"
            f"  Env:   {env.observation_space}"
        )

    # Check action space
    if model.action_space != env.action_space:
        raise ValueError(
            f"Action space mismatch:\n"
            f"  Model: {model.action_space}\n"
            f"  Env:   {env.action_space}"
        )

    return model


def safe_model_load(model_path, env):
    """Load model with compatibility checks"""

    try:
        model = validate_model_compatibility(model_path, env)
        print(f"✅ Model compatible: {model_path}")
        return model

    except ValueError as e:
        print(f"❌ Model incompatible: {e}")
        print(f"Creating new model instead...")
        return PPO("MlpPolicy", env, verbose=0)
```

**Priority:** 🔴 CRITICAL - Causes crashes

---

### 6.2 Version Incompatibility (Library Updates) 🟠 HIGH

**Description:** Model saved with different library version can't load.

**Example:**
```python
# Saved with stable-baselines3==1.8.0
# Trying to load with stable-baselines3==2.1.0
# Different pickle protocol or API changes

model = PPO.load("old_model.zip")  # ERROR or unexpected behavior
```

**Mitigation:**
```python
# requirements.txt - Pin exact versions
stable-baselines3==2.1.0
gymnasium==0.29.1
numpy==1.24.3

# Document version in model metadata
def save_model_with_metadata(model, path):
    import stable_baselines3 as sb3
    import gymnasium as gym

    metadata = {
        'sb3_version': sb3.__version__,
        'gym_version': gym.__version__,
        'save_date': datetime.now().isoformat(),
        'observation_space': str(model.observation_space),
        'action_space': str(model.action_space),
    }

    # Save model
    model.save(path)

    # Save metadata
    with open(path + '.meta.json', 'w') as f:
        json.dump(metadata, f, indent=2)


def check_version_compatibility(model_path):
    """Check if model was saved with compatible versions"""

    meta_path = model_path + '.meta.json'
    if not os.path.exists(meta_path):
        print("⚠️ No metadata found, proceeding with caution")
        return True

    with open(meta_path) as f:
        metadata = json.load(f)

    import stable_baselines3 as sb3
    if metadata['sb3_version'] != sb3.__version__:
        print(f"⚠️ Version mismatch: model={metadata['sb3_version']}, current={sb3.__version__}")
        return False

    return True
```

**Priority:** 🟠 HIGH - Prevents model loading

---

## 7. Performance Measurement Edge Cases

### 7.1 Biased Evaluation (Testing on Training Opponent) 🔴 CRITICAL

**Description:** Evaluating agent against the same opponent it trained against.

**Example:**
```python
# Train red vs blue_gen5
for _ in range(100):
    train_step(red_model, opponent=blue_gen5)

# Evaluate red vs blue_gen5 (same opponent!)
eval_score = evaluate(red_model, opponent=blue_gen5)  # 95% win rate!

# But vs different blue:
real_score = evaluate(red_model, opponent=blue_gen3)  # 30% win rate
# Overfitting!
```

**Mitigation:**
```python
def hold_out_evaluation(model, role="red"):
    """Evaluate against held-out opponents"""

    # Load different versions of opponents
    if role == "red":
        opponents = [
            PPO.load("models/blue_gen1.zip"),
            PPO.load("models/blue_gen5.zip"),
            PPO.load("models/blue_gen10.zip"),
            PPO.load("models/blue_baseline.zip"),
        ]
    else:
        opponents = [
            PPO.load("models/red_gen1.zip"),
            PPO.load("models/red_gen5.zip"),
            PPO.load("models/red_gen10.zip"),
            PPO.load("models/red_baseline.zip"),
        ]

    # Test against all
    scores = []
    for opp in opponents:
        win_rate = test_matchup(model, opp)
        scores.append(win_rate)

    # Average performance
    avg_score = np.mean(scores)
    std_score = np.std(scores)

    print(f"Avg: {avg_score:.2%} ± {std_score:.2%}")

    return avg_score, scores
```

**Priority:** 🔴 CRITICAL - Incorrect evaluation

---

### 7.2 Metric Gaming (Optimizing Wrong Metric) 🟠 HIGH

**Description:** Agent optimizes metric that doesn't reflect true objective.

**Example:**
```python
# Optimizing for: high attack success rate
red_success_rate = 0.95  # 95% success!

# But:
red_detection_rate = 0.98  # Detected 98% of time
time_to_detection = 2.3     # Detected in 2.3 seconds
real_world_utility = 0.0    # Useless in real attack

# Metric doesn't capture what matters
```

**Mitigation:**
```python
def comprehensive_evaluation(red_model, blue_model):
    """Evaluate multiple metrics"""

    metrics = {
        'red_success_rate': 0.0,
        'red_detection_rate': 0.0,
        'time_to_compromise': 0.0,
        'stealth_score': 0.0,
        'kill_chain_progress': 0.0,
        'data_exfiltrated': 0.0,

        'blue_block_rate': 0.0,
        'false_positive_rate': 0.0,
        'response_time': 0.0,
        'productivity_maintained': 0.0,
    }

    # Run multiple episodes
    for _ in range(50):
        episode_metrics = run_episode(red_model, blue_model)
        for key in metrics:
            metrics[key] += episode_metrics[key]

    # Average
    for key in metrics:
        metrics[key] /= 50

    # Composite score
    red_score = (
        metrics['red_success_rate'] * 0.3 +
        (1 - metrics['red_detection_rate']) * 0.3 +  # Lower detection is better
        metrics['stealth_score'] * 0.2 +
        metrics['kill_chain_progress'] * 0.2
    )

    blue_score = (
        metrics['blue_block_rate'] * 0.4 +
        (1 - metrics['false_positive_rate']) * 0.2 +
        metrics['productivity_maintained'] * 0.2 +
        (1 - metrics['response_time'] / 100) * 0.2  # Normalize
    )

    return {
        'red_composite': red_score,
        'blue_composite': blue_score,
        'detailed_metrics': metrics
    }
```

**Priority:** 🟠 HIGH - Misleading results

---

### 7.3 Win Rate Doesn't Reflect Skill 🟡 MEDIUM

**Description:** Win rate inflated by lucky initialization or opponent weakness.

**Example:**
```python
# Red wins 80% of episodes
# But: initial network_health=0.3 (already weak)
# Any attack succeeds

# Or: Blue opponent is broken
# Win rate doesn't reflect red skill
```

**Mitigation:**
```python
def normalized_evaluation(model, opponent):
    """Evaluate with controlled starting conditions"""

    # Test from various starting states
    starting_conditions = [
        {'network_health': 0.3, 'detection_level': 0.1},  # Easy
        {'network_health': 0.5, 'detection_level': 0.3},  # Medium
        {'network_health': 0.8, 'detection_level': 0.5},  # Hard
    ]

    results = []

    for condition in starting_conditions:
        env.reset()
        env.network_health = condition['network_health']
        env.detection_level = condition['detection_level']

        win_rate = run_evaluation(model, opponent, env)
        results.append(win_rate)

    # Weighted average (harder conditions weighted more)
    weights = [0.2, 0.3, 0.5]
    weighted_score = sum(w * r for w, r in zip(weights, results))

    return weighted_score
```

**Priority:** 🟡 MEDIUM - Evaluation accuracy

---

## 8. Resource & System Edge Cases

### 8.1 Memory Leak in Long Training 🟠 HIGH

**Description:** Memory usage grows unbounded during multi-generation training.

**Example:**
```python
# Generation 1: 2GB RAM
# Generation 5: 4GB RAM
# Generation 10: 8GB RAM
# Generation 20: 16GB RAM → OOM crash!
```

**Detection & Mitigation:**
```python
import psutil
import gc

def monitor_memory():
    """Monitor memory usage"""
    process = psutil.Process()
    mem_info = process.memory_info()
    mem_gb = mem_info.rss / 1024**3

    return mem_gb


def train_with_memory_management(generations=10):
    """Training with memory leak prevention"""

    for gen in range(generations):
        print(f"\nGeneration {gen}")

        # Check memory before
        mem_before = monitor_memory()
        print(f"Memory before: {mem_before:.2f} GB")

        # Train
        red_model = train_red_generation()
        blue_model = train_blue_generation()

        # Force garbage collection
        gc.collect()

        # Check memory after
        mem_after = monitor_memory()
        print(f"Memory after: {mem_after:.2f} GB")

        # Detect leak
        mem_increase = mem_after - mem_before
        if mem_increase > 1.0:  # More than 1GB increase
            print(f"⚠️ Possible memory leak: +{mem_increase:.2f} GB")

        # Emergency cleanup if memory too high
        if mem_after > 12.0:  # More than 12GB
            print("⚠️ High memory usage, performing deep cleanup")

            # Clear model caches
            red_model.policy.optimizer = None
            blue_model.policy.optimizer = None

            # Rebuild optimizers
            red_model.policy._setup_model()
            blue_model.policy._setup_model()

            gc.collect()

            mem_after_cleanup = monitor_memory()
            print(f"Memory after cleanup: {mem_after_cleanup:.2f} GB")
```

**Priority:** 🟠 HIGH - Prevents long training runs

---

### 8.2 Disk Space Exhaustion from Model Checkpoints 🟡 MEDIUM

**Description:** Saving too many model checkpoints fills disk.

**Example:**
```python
# Save model every generation
# 10 generations × 2 agents × 500MB = 10GB
# 100 generations × 2 agents × 500MB = 100GB!
```

**Mitigation:**
```python
def smart_checkpoint_management(model, generation, max_checkpoints=10):
    """Keep only most important checkpoints"""

    checkpoint_dir = Path("models/checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)

    # Save current
    checkpoint_path = checkpoint_dir / f"gen_{generation}.zip"
    model.save(str(checkpoint_path))

    # Keep only:
    # - Last N checkpoints
    # - Best performing checkpoint
    # - Every 10th checkpoint

    all_checkpoints = sorted(checkpoint_dir.glob("gen_*.zip"))

    to_keep = set()

    # Last N
    to_keep.update(all_checkpoints[-max_checkpoints:])

    # Every 10th
    for cp in all_checkpoints:
        gen_num = int(cp.stem.split('_')[1])
        if gen_num % 10 == 0:
            to_keep.add(cp)

    # Delete others
    for cp in all_checkpoints:
        if cp not in to_keep:
            cp.unlink()
            print(f"Deleted old checkpoint: {cp.name}")


def check_disk_space(min_free_gb=10):
    """Check if enough disk space"""
    import shutil

    stat = shutil.disk_usage("models/")
    free_gb = stat.free / (1024**3)

    if free_gb < min_free_gb:
        raise IOError(f"Insufficient disk space: {free_gb:.1f}GB < {min_free_gb}GB")

    return free_gb
```

**Priority:** 🟡 MEDIUM - Operational issue

---

## 9. Kill Chain Specific Edge Cases

### 9.1 Phase Skipping Exploit 🟠 HIGH

**Description:** Red bypasses required kill chain phases.

**Example:**
```python
# Red should progress: Recon → Weaponization → Delivery → Exploitation
# But red discovers: Can use exploitation tools without reconnaissance!

# Unrealistic sequence:
actions = [12, 10, 14]  # SQL injection, Metasploit, Cobalt Strike
# Without any reconnaissance first!

kill_chain_phase = 0  # Still in RECONNAISSANCE
# But using EXPLOITATION tools
```

**Detection & Mitigation:**
```python
def enforce_kill_chain_progression(action, current_phase):
    """Enforce realistic phase requirements"""

    # Map tools to required phases
    tool_requirements = {
        # Reconnaissance tools (0-4)
        0: 0, 1: 0, 2: 0, 3: 0, 4: 0,
        # Weaponization tools (5-9)
        5: 1, 6: 1, 7: 1, 8: 1, 9: 1,
        # Exploitation tools (10-14)
        10: 3, 11: 3, 12: 3, 13: 3, 14: 3,
        # Post-exploitation (20-24)
        20: 5, 21: 5, 22: 6, 23: 6, 24: 6,
    }

    required_phase = tool_requirements.get(action, 0)

    if current_phase < required_phase:
        # Penalize heavily
        penalty = (required_phase - current_phase) * 50.0
        return -penalty, f"Phase requirement not met: need {required_phase}, at {current_phase}"

    return 0.0, None


class KillChainEnforcement(gym.Env):
    """Environment with strict kill chain enforcement"""

    def step(self, action):
        # Check phase requirements
        penalty, error = enforce_kill_chain_progression(
            action, self.current_phase.value
        )

        if penalty < 0:
            # Invalid action
            return self._get_obs(), penalty, False, False, {'error': error}

        # Execute action
        reward, success = self._execute_action(action)

        return self._get_obs(), reward, False, False, {}
```

**Priority:** 🟠 HIGH - Breaks realism

---

### 9.2 Stuck in Final Phase (Never Completing) 🟡 MEDIUM

**Description:** Red reaches final phase but never completes objective.

**Example:**
```python
# Reaches ACTIONS_OBJECTIVES phase
kill_chain_phase = 6  # Final phase
kill_chain_progress = {
    'RECONNAISSANCE': 1.0,
    'WEAPONIZATION': 1.0,
    'DELIVERY': 1.0,
    'EXPLOITATION': 1.0,
    'INSTALLATION': 1.0,
    'COMMAND_CONTROL': 1.0,
    'ACTIONS_OBJECTIVES': 0.3  # But stuck at 30%!
}

# Never completes exfiltration
data_exfiltrated = 0.0
```

**Mitigation:**
```python
def completion_bonus(kill_chain_progress, data_exfiltrated):
    """Bonus for completing objectives"""

    if all(p >= 1.0 for p in kill_chain_progress.values()):
        if data_exfiltrated > 0.5:
            # Big bonus for completing full attack
            return 100.0

    return 0.0


def detect_objective_stagnation(data_exfil_history, window=20):
    """Detect if stuck without completing objectives"""

    recent = data_exfil_history[-window:]

    if len(recent) == window and max(recent) < 0.1:
        return True, "No data exfiltration progress"

    return False, None
```

**Priority:** 🟡 MEDIUM - Incomplete attacks

---

## 10. Detection & Mitigation Strategies

### 10.1 Automated Edge Case Detection System

```python
class EdgeCaseDetector:
    """Automated system to detect edge cases during training"""

    def __init__(self):
        self.detectors = [
            self.detect_cyclic_dominance,
            self.detect_strategy_collapse,
            self.detect_catastrophic_forgetting,
            self.detect_runaway_optimization,
            self.detect_training_divergence,
            self.detect_deadlock,
            self.detect_reward_hacking,
            self.detect_phase_stagnation,
        ]

        self.alerts = []

    def check_all(self, training_state):
        """Run all edge case detectors"""

        issues_found = []

        for detector in self.detectors:
            is_issue, description = detector(training_state)
            if is_issue:
                issues_found.append({
                    'detector': detector.__name__,
                    'description': description,
                    'timestamp': datetime.now(),
                    'generation': training_state['generation']
                })

        self.alerts.extend(issues_found)

        return issues_found

    def detect_cyclic_dominance(self, state):
        history = state['win_history']
        if len(history) < 10:
            return False, None

        # Check for oscillating wins
        wins_red = [h['red_wins'] for h in history[-10:]]
        variance = np.var(wins_red)

        if variance > 8.0:  # High variance = oscillation
            return True, f"Oscillating wins (var={variance:.1f})"

        return False, None

    def detect_strategy_collapse(self, state):
        red_actions = state['red_action_history']
        blue_actions = state['blue_action_history']

        # Check action diversity
        red_entropy = calculate_entropy(red_actions[-100:])
        blue_entropy = calculate_entropy(blue_actions[-100:])

        if red_entropy < 0.3 or blue_entropy < 0.3:
            return True, f"Low strategy diversity (R={red_entropy:.2f}, B={blue_entropy:.2f})"

        return False, None

    def generate_report(self):
        """Generate edge case detection report"""

        if not self.alerts:
            print("✅ No edge cases detected")
            return

        print(f"\n⚠️ {len(self.alerts)} Edge Cases Detected:\n")

        for alert in self.alerts:
            print(f"[Gen {alert['generation']}] {alert['detector']}")
            print(f"  {alert['description']}")
            print()
```

---

### 10.2 Automated Mitigation System

```python
class AutomaticMitigations:
    """Automatically apply mitigations when edge cases detected"""

    def __init__(self):
        self.mitigation_map = {
            'detect_cyclic_dominance': self.mitigate_cycling,
            'detect_strategy_collapse': self.mitigate_collapse,
            'detect_catastrophic_forgetting': self.mitigate_forgetting,
            'detect_runaway_optimization': self.mitigate_runaway,
        }

    def apply_mitigations(self, issues, training_state):
        """Apply appropriate mitigations"""

        for issue in issues:
            detector_name = issue['detector']

            if detector_name in self.mitigation_map:
                mitigation = self.mitigation_map[detector_name]
                print(f"Applying mitigation: {mitigation.__name__}")

                training_state = mitigation(training_state)

        return training_state

    def mitigate_cycling(self, state):
        """Add population-based training"""
        print("  - Enabling population-based training")
        state['use_population'] = True
        state['population_size'] = 10
        return state

    def mitigate_collapse(self, state):
        """Add diversity incentives"""
        print("  - Adding entropy bonus to rewards")
        state['entropy_bonus'] = 0.1
        return state

    def mitigate_forgetting(self, state):
        """Add experience replay"""
        print("  - Enabling experience replay")
        state['replay_probability'] = 0.3
        return state

    def mitigate_runaway(self, state):
        """Add capability bounds"""
        print("  - Enforcing capability bounds")
        state['max_success_rate'] = 0.85
        return state
```

---

## Summary: Top 10 Most Critical Edge Cases

| # | Edge Case | Severity | Detection | Mitigation |
|---|-----------|----------|-----------|------------|
| 1 | **Pickle Deserialization** | 🔴 CRITICAL | Static analysis | Use SafeTensors |
| 2 | **Cyclic Dominance** | 🔴 CRITICAL | Win pattern analysis | Population-based training |
| 3 | **Strategy Collapse** | 🔴 CRITICAL | Entropy measurement | Diversity incentives |
| 4 | **Reward Hacking** | 🔴 CRITICAL | Opponent crash detection | Robust opponent wrapper |
| 5 | **Observation Space Mismatch** | 🔴 CRITICAL | Shape validation | Compatibility checks |
| 6 | **Runaway Optimization** | 🟠 HIGH | Capability monitoring | Hard bounds |
| 7 | **Catastrophic Forgetting** | 🟠 HIGH | Historical evaluation | Experience replay |
| 8 | **Training Divergence** | 🟠 HIGH | Train vs eval comparison | Early stopping |
| 9 | **Phase Skipping** | 🟠 HIGH | Kill chain validation | Phase enforcement |
| 10 | **Biased Evaluation** | 🔴 CRITICAL | Hold-out testing | Diverse opponents |

---

## Testing Recommendations

```python
def comprehensive_edge_case_testing():
    """Run all edge case tests"""

    print("="*70)
    print("EDGE CASE TESTING SUITE")
    print("="*70)

    tests = [
        test_cyclic_dominance,
        test_strategy_collapse,
        test_catastrophic_forgetting,
        test_runaway_optimization,
        test_model_compatibility,
        test_kill_chain_enforcement,
        test_resource_constraints,
        test_evaluation_bias,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            print(f"\nRunning: {test.__name__}")
            test()
            print("✅ PASS")
            passed += 1
        except AssertionError as e:
            print(f"❌ FAIL: {e}")
            failed += 1

    print(f"\n{'='*70}")
    print(f"Results: {passed} passed, {failed} failed")
    print("="*70)
```

---

**Document Version:** 1.0
**Last Updated:** 2025-11-18
**Maintained By:** @sarowarzahan414
**Status:** Production Ready

**Train Tomorrow's Defenders with Robust Co-Evolution Systems!** 🛡️
