# ACEAC: Adversarial Co-Evolution for Autonomous Cybersecurity

## 📋 Table of Contents
- [Project Overview](#project-overview)
- [Current State Analysis](#current-state-analysis)
- [Novelty Assessment](#novelty-assessment)
- [Roadmap to Novel Research](#roadmap-to-novel-research)
- [Implementation Guide](#implementation-guide)
- [Oracle Free Tier Setup](#oracle-free-tier-setup)
- [Expected Contributions](#expected-contributions)
- [Timeline](#timeline)
- [Citation](#citation)

---

## 🎯 Project Overview

**ACEAC (Adversarial Co-Evolution for Autonomous Cybersecurity)** is a research project implementing adversarial co-evolutionary AI for penetration testing and defensive cybersecurity.

### Current Features (v1 & v2)

**Version 1.0 (Baseline):**
- 20-dimensional observation space
- 10 actions per agent (Red/Blue)
- Basic Red vs Blue co-evolution training
- PPO (Proximal Policy Optimization) algorithm
- Simple gym environment

**Version 2.0 (Advanced):**
- 62-dimensional observation space
- 25 real-world tools per agent (50 total)
- Cyber Kill Chain implementation (7 phases)
- SWAP RL (Self-Play with Adaptive Policies)
- Realistic tool simulation (Metasploit, Nmap, Snort, etc.)

### File Structure

```
ACEAC-local/
├── validate_aceac_agents.py          # Agent comparison & validation
├── aceac_v2_visualize.py             # Training visualization suite
├── aceac_v2_validate.py              # v2 validation script
├── aceac_v2_swap_rl.py               # SWAP RL training
├── aceac_v2_cyber_killchain.py       # Cyber Kill Chain environment
├── aceac_red_100ep_validate_FIXED.py # Red agent baseline (100 ep)
├── aceac_blue_100ep_validate_FIXED.py # Blue agent baseline (100 ep)
├── aceac_cyber_range.py              # Red team environment
├── aceac_blue_team_range.py          # Blue team environment
├── aceac_coevolution_FIXED.py        # Co-evolution training
├── models/                           # Trained model checkpoints
├── logs/                             # Training logs (JSON)
└── README.md                         # This file
```

---

## 🔍 Current State Analysis

### Strengths

**✅ Well-designed Architecture:**
- Proper Gymnasium API implementation
- Type safety with numpy array → int conversions
- Clear separation of concerns
- Comprehensive logging (JSON format)

**✅ Realistic Cybersecurity Simulation:**
- Real tool names (Metasploit, Nmap, Burp Suite, Snort, Wireshark, etc.)
- Cyber Kill Chain phases: Reconnaissance → Weaponization → Delivery → Exploitation → Installation → Command & Control → Actions on Objectives
- Stealth penalties for noisy attacks
- Detection mechanics

**✅ Research-Oriented:**
- Visualization capabilities (matplotlib graphs)
- Performance metrics tracking
- Baseline vs co-evolved agent comparison

### Critical Issues

**❌ Code Duplication:**
- `aceac_cyber_range.py` and `aceac_blue_team_range.py` are identical files
- Both implement Red team environment despite naming

**❌ Inconsistent Observation Spaces:**
- v1: 20D observation space
- v2: 62D observation space
- Co-evolution scripts need compatibility checks

**❌ Hardcoded Values:**
```python
# Dates (future/placeholder dates)
"date": "2025-10-08"

# User information
"user": "sarowarzahan414"

# Location
"location": "Kali Linux VirtualBox"

# Magic numbers throughout
attack_power = 0.08 + action * 0.015  # No configuration
reward -= 2.0  # Hardcoded penalties
```

**❌ Environment Realism:**
```python
# Random observations (defeats learning)
obs[3:] = np.random.random(17) * 0.5

# Single scalar for entire network
self.network_security = 0.8

# Purely probabilistic (no skill)
success_prob = attack_power * (1.2 - self.network_security)
```

**❌ SWAP RL Implementation:**
```python
# Random sampling (no curriculum, no diversity metrics)
def sample_opponent(self):
    idx = np.random.randint(0, len(self.policies))
    return self.policies[idx]
```

### Code Quality Issues

- String concatenation instead of f-strings
- Magic numbers everywhere
- No configuration files
- No unit tests
- No docstrings for key methods
- No hyperparameter tuning
- No reproducibility (inconsistent seed setting)

---

## 🎓 Novelty Assessment

### Current Novelty Score: **2/10**

### What is NOT Novel

**❌ Co-evolutionary Training**
- Red vs Blue co-evolution exists in game theory and multi-agent RL
- Reference: "Emergent Complexity via Multi-Agent Competition" (OpenAI, 2017)
- Current implementation: Basic alternating training, no innovation

**❌ PPO for Cybersecurity**
- Well-established in literature
- No novel algorithm modifications

**❌ Cyber Kill Chain**
- Framework from Lockheed Martin (2011)
- Linear progression model is overly simplistic

**❌ SWAP RL**
- Essentially Fictitious Self-Play or PSRO (Policy Space Response Oracles)
- Reference: "A Unified Game-Theoretic Approach to Multiagent RL" (Lanctot et al., 2017)
- Name appears to be rebranding existing techniques

### What COULD Be Novel (Not Implemented Yet)

**🤔 Quality-Diversity for Cybersecurity**
- First application of MAP-Elites/CVT-MAP-Elites to adversarial cybersecurity
- Behavioral embeddings for attack/defense strategies
- **Requires: Diversity metrics, behavioral characterization**

**🤔 Realistic Multi-Host Environment**
- Network topology with multiple hosts
- Tool preconditions and dependencies
- Attack graph generation
- **Requires: Graph-based environment, realistic CVEs**

**🤔 CTF Transfer Learning**
- First RL agent validated on real CTF challenges
- Transfer from simulation to HackTheBox/TryHackMe
- **Requires: Integration with real vulnerable systems**

---

## 🚀 Roadmap to Novel Research

Three paths to achieve genuine novelty in **3-4 months**:

### Option A: Practical/Applied Research (RECOMMENDED)

**Timeline: 3-4 months**
**Publication Target: USENIX Security, IEEE S&P, ACM CCS**

#### Phase 1: Quality-Diversity Policy Pools (3-4 weeks)
Replace random sampling with behavioral diversity metrics.

**Key Innovation:**
- MAP-Elites archive for cybersecurity strategies
- Behavioral descriptors: (aggression, stealth, speed, tool diversity)
- Coverage metrics for strategy space

**Expected Result:** 60-80% behavioral coverage vs 10-20% with random sampling

#### Phase 2: Realistic Multi-Host Environment (4-6 weeks)

**Key Innovation:**
- Multi-host network topology (DMZ, Internal, Critical zones)
- Tool dependency graph (can't lateral-move without initial access)
- Attack graph generation
- Real CVE database integration

**Expected Result:** Environment complexity matches real enterprise networks

#### Phase 3: CTF Validation (2-3 weeks)

**Key Innovation:**
- Systematic benchmark (5+ real CTF scenarios)
- Human expert baseline comparisons
- Transfer learning validation

**Expected Result:** Agent achieves >40% success on easy CTFs, >20% on medium

#### Expected Novel Claims

**After completing Option A:**
1. ✅ "First quality-diversity approach for adversarial cybersecurity AI"
2. ✅ "Most realistic RL environment for penetration testing with multi-host topology"
3. ✅ "First RL agent validated on real-world CTF challenges with human baselines"
4. ✅ "Systematic benchmark for cybersecurity co-evolution research"

---

### Option B: Theoretical/Algorithmic Research

**Timeline: 4-5 months**
**Publication Target: NeurIPS, ICML, ICLR**

#### Phase 1: Hierarchical SWAP RL (6-8 weeks)

**Key Innovation:**
- Three-level decision hierarchy:
  - Strategic: Kill chain phase selection
  - Tactical: Tool category selection
  - Execution: Specific tool selection
- Options framework with temporal abstraction

**Expected Result:** 30-40% faster learning vs flat policy

#### Phase 2: Convergence Analysis (4-6 weeks)

**Theoretical Contributions:**
- Prove convergence to Nash equilibrium (or approximate)
- Sample complexity bounds for policy pool
- Regret bounds for exploration-exploitation

**Expected Result:** Formal convergence guarantees

#### Phase 3: Interpretable Strategy Extraction (3-4 weeks)

**Key Innovation:**
- Policy distillation to decision trees
- Attention mechanisms for tool selection
- Human-readable playbook generation

**Expected Result:** 85%+ accuracy in predicting agent actions with decision tree

---

### Option C: Hybrid Path (BEST FOR PUBLICATION) ⭐

**Timeline: 3 months**
**Publication Target: Top-tier security or AI conference**

Combines ONE algorithmic + ONE empirical + ONE environment improvement.

**Selected Components:**
1. Quality-Diversity Policy Pools (Algorithmic)
2. Multi-Host Network with Tool Dependencies (Environment)
3. CTF Benchmark Suite + Baselines (Validation)

**Why This Works:**
- Balanced contribution (theory + practice)
- Achievable in 3 months
- Strong empirical validation
- Multiple novelty claims

---

## 💻 Implementation Guide

### Prerequisites

```bash
# Python 3.8+
python3 --version

# Core dependencies
pip install gymnasium
pip install stable-baselines3[extra]
pip install numpy matplotlib pandas
pip install torch  # CPU version is fine
pip install networkx  # For graph-based environment
```

### Week-by-Week Plan

#### Week 1-2: Code Refactoring

**Fix current issues:**

```bash
# 1. Create configuration file
cat > config.yaml << EOF
training:
  episodes_per_generation: 100
  num_generations: 10
  learning_rate: 3e-4
  n_steps: 2048
  batch_size: 64

environment:
  max_steps: 100
  initial_security: 0.8
  detection_threshold: 0.9

tools:
  red_tools: 25
  blue_tools: 25
EOF

# 2. Create constants file
cat > constants.py << 'EOF'
"""Configuration constants"""

# Tool IDs
OFFENSIVE_TOOLS = {
    'NMAP': 0, 'MASSCAN': 1, 'METASPLOIT': 10,
    'SQLMAP': 12, 'HYDRA': 15
}

DEFENSIVE_TOOLS = {
    'SNORT': 0, 'SURICATA': 1, 'SPLUNK': 12,
    'WAZUH': 11, 'FAIL2BAN': 16
}

# Reward constants
REWARD_SUCCESS = 15.0
REWARD_FAILURE = -8.0
STEALTH_PENALTY = -2.0
EOF
```

#### Week 3-6: Quality-Diversity Implementation

Create `aceac_quality_diversity.py`:

```python
"""
Quality-Diversity Training for ACEAC
Implements MAP-Elites for behavioral diversity in cybersecurity strategies
"""

import numpy as np
from stable_baselines3 import PPO
from collections import defaultdict
import pickle
import json

class BehavioralDescriptor:
    """Extract behavioral characteristics from policy"""

    @staticmethod
    def compute_behavior(policy, env, num_episodes=10):
        """
        Behavioral dimensions:
        - Aggression: damage_dealt / max_possible
        - Stealth: 1 - (detections / actions)
        - Speed: 1 - (time_taken / max_time)
        - Diversity: unique_tools / total_tools
        """
        total_damage = 0
        total_detections = 0
        total_actions = 0
        total_time = 0
        tool_usage = set()

        for _ in range(num_episodes):
            obs, _ = env.reset()
            done = False
            step = 0

            while not done and step < env.max_steps:
                action, _ = policy.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = env.step(action)

                total_damage += info.get('damage_dealt', 0)
                total_detections += info.get('detected', 0)
                total_actions += 1
                tool_usage.add(int(action))
                step += 1

            total_time += step

        # Normalize to [0, 1]
        aggression = total_damage / (num_episodes * env.max_damage)
        stealth = 1.0 - (total_detections / total_actions) if total_actions > 0 else 0
        speed = 1.0 - (total_time / (num_episodes * env.max_steps))
        diversity = len(tool_usage) / env.action_space.n

        return {
            'aggression': np.clip(aggression, 0, 1),
            'stealth': np.clip(stealth, 0, 1),
            'speed': np.clip(speed, 0, 1),
            'diversity': np.clip(diversity, 0, 1)
        }


class MAPElitesArchive:
    """MAP-Elites archive for quality-diversity"""

    def __init__(self, grid_size=20, dimensions=['aggression', 'stealth']):
        self.grid_size = grid_size
        self.dimensions = dimensions
        self.archive = {}  # (x, y) -> PolicyEntry

    def discretize_behavior(self, behavior):
        """Map continuous behavior to grid cell"""
        coords = []
        for dim in self.dimensions:
            value = behavior[dim]
            cell = int(value * self.grid_size)
            cell = min(cell, self.grid_size - 1)
            coords.append(cell)
        return tuple(coords)

    def add(self, policy_params, fitness, behavior):
        """Add policy to archive if it improves cell"""
        cell = self.discretize_behavior(behavior)

        if cell not in self.archive or fitness > self.archive[cell]['fitness']:
            self.archive[cell] = {
                'params': policy_params,
                'fitness': fitness,
                'behavior': behavior
            }
            return True
        return False

    def sample(self):
        """Sample random policy from archive"""
        if not self.archive:
            return None
        cell = np.random.choice(list(self.archive.keys()))
        return self.archive[cell]

    def get_coverage(self):
        """Percentage of cells filled"""
        total_cells = self.grid_size ** len(self.dimensions)
        return len(self.archive) / total_cells

    def get_stats(self):
        """Archive statistics"""
        if not self.archive:
            return {'size': 0, 'coverage': 0, 'avg_fitness': 0, 'max_fitness': 0}

        fitnesses = [entry['fitness'] for entry in self.archive.values()]

        return {
            'size': len(self.archive),
            'coverage': self.get_coverage(),
            'avg_fitness': np.mean(fitnesses),
            'max_fitness': np.max(fitnesses),
            'min_fitness': np.min(fitnesses)
        }


# Example usage
if __name__ == "__main__":
    from aceac_v2_cyber_killchain import ACEACv2Environment

    def env_fn(agent_role):
        return ACEACv2Environment(agent_role=agent_role)

    # Create archive
    archive = MAPElitesArchive(grid_size=20)

    # Train and add policies
    env = env_fn('red')
    for gen in range(10):
        policy = PPO('MlpPolicy', env, verbose=0)
        policy.learn(total_timesteps=5000)

        # Evaluate
        behavior = BehavioralDescriptor.compute_behavior(policy, env)
        fitness = np.random.random()  # Placeholder

        # Add to archive
        added = archive.add(policy.get_parameters(), fitness, behavior)

        print(f"Gen {gen}: Coverage={archive.get_coverage():.2%}, Added={added}")

    print(f"\nFinal stats: {archive.get_stats()}")
```

#### Week 7-10: Realistic Multi-Host Environment

See implementation in the main guide above.

#### Week 11-12: CTF Validation

See CTF benchmark implementation in the main guide above.

---

## ☁️ Oracle Free Tier Setup

### What You Get (Always Free)

```
Compute:
└── Ampere A1: 4 ARM cores + 24 GB RAM ⭐

Storage:
└── 200 GB Block Storage

Network:
└── 10 TB/month outbound transfer
```

### Complete Setup

```bash
# 1. Create Oracle Cloud account
# https://www.oracle.com/cloud/free/

# 2. Create Ampere A1 instance
#    Shape: VM.Standard.A1.Flex
#    OCPUs: 4 (max free)
#    RAM: 24 GB (max free)
#    OS: Ubuntu 22.04 Minimal
#    Boot: 50 GB

# 3. SSH into instance
ssh -i ~/.ssh/oracle_key ubuntu@<PUBLIC_IP>

# 4. Run setup script
curl -O https://raw.githubusercontent.com/sarowarzahan414/ACEAC-local/main/setup_oracle.sh
chmod +x setup_oracle.sh
./setup_oracle.sh
```

**Setup Script: `setup_oracle.sh`**

```bash
#!/bin/bash
# ACEAC Oracle Free Tier Setup

echo "ACEAC Setup for Oracle Free Tier"
echo "================================="

# Update system
sudo apt update && sudo apt upgrade -y

# Install dependencies
sudo apt install -y \
    python3.11 python3-pip \
    git tmux htop \
    docker.io \
    build-essential

# Python packages
pip3 install --upgrade pip
pip3 install \
    gymnasium \
    'stable-baselines3[extra]' \
    numpy matplotlib pandas \
    networkx \
    torch --index-url https://download.pytorch.org/whl/cpu

# Clone repository
git clone https://github.com/sarowarzahan414/ACEAC-local.git
cd ACEAC-local

# Create directories
mkdir -p models/{baseline,quality_diversity,realistic,checkpoints}
mkdir -p logs/{training,validation,ctf}
mkdir -p data/{ctf_scenarios,benchmarks}

# Memory optimization
echo "export MALLOC_TRIM_THRESHOLD_=100000" >> ~/.bashrc
echo "export PYTHONHASHSEED=42" >> ~/.bashrc

echo ""
echo "Setup complete!"
echo "Start training: python3 aceac_quality_diversity.py"
```

### Resource Usage

| Phase | Memory | CPU | Storage | Time | Cost |
|-------|--------|-----|---------|------|------|
| Quality-Diversity | 3 GB | 2 cores | 10 GB | 2 weeks | $0 |
| Realistic Env | 2 GB | 2 cores | 5 GB | 2 weeks | $0 |
| CTF Validation | 6 GB | 3 cores | 30 GB | 1 week | $0 |
| **Total** | **6 GB** | **3 cores** | **45 GB** | **5 weeks** | **$0** |

**Oracle provides: 24 GB RAM, 4 cores, 200 GB storage**
**Usage: 25% RAM, 75% CPU, 22% storage**

✅ **Plenty of headroom for experiments!**

---

## 🎓 Expected Contributions

After completing Option C, you can claim:

### 1. Algorithmic Contribution

**"First quality-diversity approach for adversarial cybersecurity AI"**

- Novel behavioral embeddings for attack/defense strategies
- MAP-Elites archive for cybersecurity policy pools
- Coverage metrics for strategy space exploration

### 2. System Contribution

**"Most realistic RL environment for penetration testing"**

- Multi-host network topology with security zones
- Tool dependency modeling and attack graphs
- Realistic CVE-based vulnerability simulation

### 3. Empirical Contribution

**"First RL agent validated on real CTF challenges"**

- Systematic benchmark suite (5+ scenarios)
- Human expert baseline comparisons
- Transfer learning from simulation to real systems

### 4. Community Contribution

**"Open-source benchmark for cybersecurity co-evolution research"**

- Reproducible evaluation framework
- Comprehensive baseline implementations
- Public leaderboard for research comparison

---

## 📅 Timeline

### 3-Month Implementation Plan

```
Month 1: Foundation
├── Week 1-2: Code refactoring, fix issues
├── Week 3-4: Quality-Diversity implementation
└── Deliverable: QD training working

Month 2: Realism
├── Week 5-8: Multi-host environment
├── Week 9: Network topology + tool dependencies
└── Deliverable: Realistic environment working

Month 3: Validation
├── Week 10: CTF scenario setup
├── Week 11: Benchmark experiments
├── Week 12: Analysis + paper writing
└── Deliverable: Paper draft ready

Total: 12 weeks (3 months)
```

### Milestones

- ✅ **Week 4:** Quality-Diversity working, behavioral embeddings computed
- ✅ **Week 8:** Multi-host environment, attack graphs generated
- ✅ **Week 10:** CTF scenarios running, agent tested on real VMs
- ✅ **Week 12:** Paper submitted to conference

---

## 📊 Publication Strategy

### Target Venues

**Top-tier Security:**
- USENIX Security Symposium (Deadline: Usually February/June)
- IEEE Symposium on Security and Privacy (S&P) (Deadline: Usually May/November)
- ACM CCS (Deadline: Usually January/May)
- NDSS (Deadline: Usually May/August)

**Top-tier AI:**
- NeurIPS (Deadline: Usually May)
- ICML (Deadline: Usually January)
- ICLR (Deadline: Usually September)

**Recommended:** USENIX Security (strong empirical work) or ACM CCS (hybrid theory/practice)

### Paper Structure

```
Title: "Adversarial Co-Evolution with Quality-Diversity for
       Autonomous Penetration Testing"

Abstract: (200 words)
├── Problem: Limited realism in cybersecurity RL
├── Solution: Quality-diversity + multi-host + CTF validation
├── Results: X% success on real CTFs, Y% coverage improvement
└── Impact: First validated autonomous pentesting system

1. Introduction
2. Background
3. Approach (Quality-Diversity + Realistic Environment)
4. Experimental Setup (Benchmarks + Baselines)
5. Results (Coverage, CTF Performance, Ablations)
6. Discussion
7. Related Work
8. Conclusion
```

---

## 📚 Citation

If you use this work, please cite:

```bibtex
@misc{aceac2025,
  title={ACEAC: Adversarial Co-Evolution for Autonomous Cybersecurity},
  author={Zahan, Sarowar},
  year={2025},
  url={https://github.com/sarowarzahan414/ACEAC-local}
}
```

---

## 🤝 Contributing

Contributions welcome! Areas for improvement:

1. Additional CTF scenarios
2. More realistic detection systems
3. Blue team enhancements
4. Visualization improvements
5. Documentation

---

## 📄 License

[Specify your license: MIT, Apache 2.0, GPL, etc.]

---

## 📧 Contact

- **Author:** Sarowar Zahan
- **GitHub:** [@sarowarzahan414](https://github.com/sarowarzahan414)

---

## 🙏 Acknowledgments

- Stable-Baselines3 for RL implementations
- Gymnasium for environment interface
- Oracle Cloud for free compute resources
- Cybersecurity research community

---

## 📝 Changelog

### v2.0 (Planned - 2025)
- Quality-Diversity policy pools
- Multi-host realistic environment
- CTF validation framework
- Comprehensive benchmarks

### v1.0 (Current)
- Basic Red vs Blue co-evolution
- Cyber Kill Chain integration
- SWAP RL implementation
- 50 real-world tools

---

## 🔗 Related Projects

- [CyberBattleSim](https://github.com/microsoft/CyberBattleSim) - Microsoft's cyber reasoning system
- [CAGE Challenge](https://github.com/cage-challenge/cage-challenge-1) - Autonomous cyber defense
- [NetworkAttackSimulator](https://github.com/Jjschwartz/NetworkAttackSimulator) - Network penetration simulation

---

**Last Updated:** 2025-01-20

**Status:** 🚧 Under Development → Novel Research Implementation

**Next Steps:** Begin Phase 1 (Quality-Diversity) on Oracle Free Tier
