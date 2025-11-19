# ACEAC: Adaptive Cyber Environment for Adversarial Co-evolution

**Status: Early Research Prototype** 🚧

**Last Updated: 2025-11-19**

---

## What This Actually Is

A reinforcement learning experiment exploring whether attack and defense agents can learn meaningful cybersecurity strategies through adversarial co-evolution.

**Key Components:**
- Abstract environment representing network security state (32-dimensional vector)
- Red agent (attacker) learns to compromise network
- Blue agent (defender) learns to secure network
- Zero-sum game: one agent's win = other agent's loss
- Co-evolutionary training: agents improve by facing increasingly skilled opponents

---

## What This Is NOT

❌ **Not production-ready** - This is research code, not a security product
❌ **Not real threat intelligence** - Abstract simulation, not actual CVEs or TTPs
❌ **Not compliance-certified** - No regulatory compliance claims
❌ **Not validated at scale** - Limited testing, early results
❌ **Not real network simulation** - Abstract state representation, not actual systems

---

## Current Status

### ✅ What Works
- [x] Zero-sum adversarial environment (balanced, no timeout advantage)
- [x] PPO-based agent training
- [x] Co-evolutionary training loop
- [x] Basic evaluation metrics (win rates, rewards)
- [x] Environment balance validation tests

### ⏳ What's In Progress
- [ ] Baseline comparisons (vs random, vs heuristics) ← **CRITICAL NEXT STEP**
- [ ] Multi-seed validation (statistical significance)
- [ ] Strategy analysis (what agents actually learned)
- [ ] Ablation studies (which components matter)

### ❌ What Doesn't Exist Yet
- [ ] Real threat intelligence integration (future work)
- [ ] Compliance framework (aspirational)
- [ ] Production deployment (not planned)
- [ ] Real-world validation (would require major redesign)

---

## Installation

### Requirements
```bash
# Python 3.11+
pip install -r requirements.txt
```

**Dependencies:**
- numpy >= 1.21.0
- gymnasium >= 0.28.0
- stable-baselines3 >= 2.0.0
- torch >= 2.0.0 (CPU version sufficient for this scale)

### Quick Start

```bash
# 1. Validate environment balance
python quick_timeout_test.py

# 2. Run quick training test (1 generation, ~2 minutes)
python aceac_zerosum_training.py --test

# 3. Validate trained agents beat baselines
python validate_training.py  # (requires trained models from step 2)
```

---

## Current Results

**Environment Balance (Validated):**
- ✅ Random vs Random: ~5-10% wins each, ~80-90% draws
- ✅ No timeout advantage for either side
- ✅ Zero-sum property verified (sum ≈ 0)

**Baseline Comparisons (PENDING VALIDATION):**
- ⏳ Trained Red vs Random Blue: **TBD**
- ⏳ Trained Red vs Greedy Blue: **TBD**
- ⏳ Trained Blue vs Random Red: **TBD**
- ⏳ Trained Blue vs Greedy Red: **TBD**

**If trained agents can't beat Random >70% and Greedy >60%, training doesn't work.**

---

## Architecture

### Environment (`aceac_zerosum_environment.py`)

**State Representation:**
- 32-dimensional continuous vector [0.0, 1.0]
- 0.0 = fully secured, 1.0 = fully compromised
- Abstract representation (not mapped to specific systems/vulnerabilities)

**Actions:**
- 25 discrete actions per agent
- Each action affects ~25% of state dimensions
- Effect size: 0.05-0.15 per dimension
- Red pushes state UP (compromise), Blue pushes state DOWN (secure)

**Win Conditions:**
- Red wins: state mean > 0.7 (decisive compromise)
- Blue wins: state mean < 0.3 (decisive defense)
- Draw: timeout without decisive victory (state 0.3-0.7)

**Rewards:**
- Win: +100
- Loss: -100
- Draw: 0
- Per-step shaping: ±0.25 max (guides learning without dominating outcome)

### Training (`aceac_zerosum_training.py`)

**Algorithm:** Proximal Policy Optimization (PPO) via Stable-Baselines3

**Co-evolution Process:**
1. Red trains against current Blue (2048 steps)
2. Blue trains against updated Red (2048 steps)
3. Evaluate both agents (test battles)
4. Repeat for N generations

**Hyperparameters:**
- Learning rate: 0.0003
- Batch size: 64
- Entropy coefficient: 0.01 (exploration)
- Gamma: 0.99 (discount factor)

---

## Validation Protocol

### Phase 1: Environment Validation ✅ **COMPLETE**

**Test:** `quick_timeout_test.py`
**Result:** All contested states (0.3-0.7) produce draws, decisive victories work correctly

**Proves:** Environment logic is correct and balanced

### Phase 2: Baseline Validation ⏳ **IN PROGRESS**

**Test:** `validate_training.py`
**Baselines:**
- Random: Sanity check (should beat >70%)
- Greedy: Simple heuristic (should beat >60%)
- Passive: Does nothing (should beat >95%)

**Proves:** Agents learned meaningful strategies (not just random behavior)

### Phase 3: Statistical Validation ❌ **NOT DONE**

**Test:** Multi-seed runs (5+ different random seeds)
**Analysis:** Mean ± std of win rates, p-values vs baselines

**Proves:** Results are reproducible and statistically significant

### Phase 4: Strategy Analysis ❌ **NOT DONE**

**Test:** Analyze action sequences, state trajectories, learned behaviors
**Analysis:** Cluster strategies, identify archetypes, explain in plain language

**Proves:** Agents learned interpretable strategies (not black-box exploitation)

---

## Known Limitations

### Technical Limitations
1. **Abstract environment** - Not mapped to real network topology, systems, or vulnerabilities
2. **Small scale** - 32-dimensional state, 25 actions (real networks have millions of states)
3. **No multi-step attacks** - Actions are independent (real attacks chain: recon → exploit → pivot → exfiltrate)
4. **No real defensive tools** - Abstract actions (real defense uses EDR, SIEM, firewalls, etc.)

### Research Limitations
1. **No baseline comparisons yet** - Haven't proven agents beat simple heuristics
2. **No generalization testing** - Unknown if strategies transfer to different environment sizes
3. **No ablation studies** - Don't know which components (curiosity, population, etc.) actually help
4. **Limited scale testing** - Haven't tested with 100+ generations or large state spaces

### Practical Limitations
1. **Not production-ready** - Research prototype, not hardened software
2. **No real-world validation** - Never tested against actual attacks/defenses
3. **No integration** - Standalone system, doesn't connect to real security tools
4. **No interpretability tools** - Hard to explain why agent chose specific action

---

## How to Interpret Results

### Good Results (Evidence of Learning)
- ✅ Trained agents beat Random >70% (better than chance)
- ✅ Trained agents beat Greedy >60% (better than simple rules)
- ✅ Draw rate decreases over training (more decisive play)
- ✅ Win rates oscillate (evidence of arms race)
- ✅ Consistent across multiple seeds (reproducible)

### Bad Results (Training Failed)
- ❌ Can't beat Random (not learning anything)
- ❌ Can't beat Greedy (not learning useful strategies)
- ❌ Draw rate stays high (not learning decisive play)
- ❌ One side dominates 100% (imbalanced environment)
- ❌ Results vary wildly across seeds (unstable training)

### What High Draw Rate Means
- **Gen 1: ~70-80% draws** = NORMAL (agents learning basics)
- **Gen 10: ~50-60% draws** = GOOD (strategies emerging)
- **Gen 20: ~30-40% draws** = EXCELLENT (decisive play)

**Draw rate should DECREASE over training.** If it stays high, agents aren't learning to win decisively.

---

## Critical Next Steps (Priority Order)

### This Week
1. **Run baseline comparisons** (`validate_training.py`) ← **MOST CRITICAL**
2. **Multi-seed validation** (run training with 5+ seeds, check consistency)
3. **Document actual results** (replace TBD with real numbers)

### This Month
4. **Ablation studies** (test with/without curiosity, population, etc.)
5. **Strategy analysis** (understand what agents learned)
6. **Statistical testing** (p-values, confidence intervals)

### Later (If Baseline Tests Pass)
7. **Scale testing** (larger environments, more generations)
8. **Generalization testing** (train on X-dim, test on Y-dim)
9. **Interpretability tools** (visualize strategies, explain decisions)

---

## Repository Structure

```
ACEAC-local/
├── aceac_zerosum_environment.py    # Core environment (balanced)
├── aceac_zerosum_training.py        # Training loop (co-evolution)
├── baseline_agents.py               # Random/Greedy/Passive baselines
├── validate_training.py             # Critical validation tests
├── quick_timeout_test.py            # Environment balance validation
├── diagnostic_balance_tests.py      # Comprehensive environment tests
│
├── ANALYSIS_TRAINING_RESULTS.md     # Root cause analysis (timeout fix)
├── VALIDATION_RESULTS.md            # Environment validation results
├── NEXT_STEPS.md                    # Action plan
│
├── models/                          # Saved models (if training run)
│   └── zerosum/
│       ├── red_gen1.zip
│       └── blue_gen1.zip
│
└── validation_results/              # Baseline comparison results
    └── baseline_comparison.json
```

**Aspirational documents (future work, not current capabilities):**
- `THREAT_INTEL_INTEGRATION_GUIDE.md` → Future work
- `COMPLIANCE_MATRIX.md` → Future work
- `DYNAMIC_VS_STATIC_COMPARISON.md` → Not implemented

---

## Contributing

This is a research prototype. Contributions should focus on:

1. **Validation** - Add more baseline comparisons, statistical tests
2. **Analysis** - Tools to understand what agents learned
3. **Documentation** - Honest documentation of what works/doesn't work

**Not accepting (yet):**
- Production features (dashboards, APIs, deployment)
- New training algorithms (until we validate current one works)
- Real-world integration (until abstract version is proven)

---

## Citation

If you use this code in research, please cite:

```
[PENDING: Research paper in progress]

Meanwhile, cite this repository:
@software{aceac2025,
  author = {[Your Name]},
  title = {ACEAC: Adaptive Cyber Environment for Adversarial Co-evolution},
  year = {2025},
  url = {https://github.com/sarowarzahan414/ACEAC-local},
  note = {Early research prototype}
}
```

---

## License

[Specify your license here]

---

## Acknowledgments

- Built with Stable-Baselines3 (PPO implementation)
- Environment based on Gymnasium framework
- Timeout fix analysis by Claude (Anthropic)

---

## Honesty Statement

**This README describes the system AS IT IS, not as we hope it will be.**

- ✅ Claims about environment balance: **Validated**
- ⏳ Claims about agent learning: **Pending validation**
- ❌ Claims about real-world applicability: **Not tested**

**If validation tests fail, we will update this README to reflect that.**

Research requires honesty. We're committed to reporting what actually works, not what we wish worked.

---

**Questions? Check [NEXT_STEPS.md](NEXT_STEPS.md) for current status and action plan.**
