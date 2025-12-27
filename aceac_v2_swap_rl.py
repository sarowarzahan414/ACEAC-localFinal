"""
ACEAC v2.0 - Enhanced QD-SWAP RL Training (IMPROVEMENTS A + B + C + D)
Self-Play with Adaptive Policies + Quality-Diversity

ENHANCEMENTS:
  A. Increased Grid Resolution (10x10 → 20x20)
  B. 3rd Behavioral Dimension (success rate added)
  C. Train Longer (10 → 20 generations)
  D. More Episodes Per Generation (100 → 200)

INNOVATIONS:
  #1: Quality-Diversity Policy Pool (MAP-Elites)
  #2: Enhanced Behavioral Characterization (3D)

Author: @sarowarzahan414
Date: 2025-12-08 (Enhanced)

FIXED: 2025-12-25 (Blue behavioral descriptor bug)
>>>>>>> 88d0ae3 (Syncing local files with GitHub)
"""

import numpy as np
from stable_baselines3 import PPO
from aceac_v2_cyber_killchain import ACEACv2Environment
import json
import time
from datetime import datetime, timezone
import os
import pickle
from collections import defaultdict
import random


# ============================================================================
# CONFIGURATION
# ============================================================================

class QDConfig:
    """Configuration for QD-SWAP training"""
    
    # Grid Resolution (IMPROVEMENT A)
    GRID_RESOLUTION = 20  # Changed from 10 to 20 (20x20x20 = 8000 cells)
    
    # Behavioral Dimensions (IMPROVEMENT B)
    BEHAVIOR_DIMS = 3  # Changed from 2 to 3 (added success rate)
    
    # Training Parameters (IMPROVEMENTS C + D)
    NUM_GENERATIONS = 20  # Changed from 10 to 20 (train longer)
    EPISODES_PER_GEN = 200  # Changed from 100 to 200 (more exploration)
    
    # Pool Settings
    POOL_SIZE = 10  # Increased to accommodate more policies
    MAX_ARCHIVE_SIZE = 200  # Increased for larger grid
    
    # Evaluation
    BEHAVIOR_EVAL_EPISODES = 5
    PERFORMANCE_EVAL_EPISODES = 5
    
    def __str__(self):
        return f"""QD Configuration:
  Grid Resolution: {self.GRID_RESOLUTION}x{self.GRID_RESOLUTION}x{self.GRID_RESOLUTION} = {self.GRID_RESOLUTION**self.BEHAVIOR_DIMS:,} cells
  Behavior Dimensions: {self.BEHAVIOR_DIMS}D
  Generations: {self.NUM_GENERATIONS}
  Episodes per Gen: {self.EPISODES_PER_GEN}
  Total Episodes: {self.NUM_GENERATIONS * self.EPISODES_PER_GEN * 2:,}"""


# ============================================================================
# INNOVATION #1: Quality-Diversity Policy Pool
# ============================================================================

class PolicyCell:
    """Container for policy with fitness and behavioral metadata"""
    def __init__(self, policy, fitness, behavior, generation):
        self.policy = policy
        self.fitness = fitness
        self.behavior = behavior
        self.generation = generation
        self.creation_time = time.time()
    
    def __repr__(self):
        return f"PolicyCell(fitness={self.fitness:.2f}, behavior={self.behavior}, gen={self.generation})"


class EnhancedQualityDiversityPool:
    """
    INNOVATION #1: Enhanced Quality-Diversity Pool
    
    IMPROVEMENTS:
    - Higher resolution grid (20x20x20)
    - 3D behavioral space
    - Better memory management
    - Detailed statistics tracking
    """
    
    def __init__(self, pool_size=10, agent_type="red", 
                 behavior_dims=3, resolution=20, max_archive_size=200):
        self.pool_size = pool_size
        self.agent_type = agent_type
        self.behavior_dims = behavior_dims
        self.resolution = resolution
        self.max_archive_size = max_archive_size
        
        # MAP-Elites archive
        self.archive = {}
        
        # Statistics
        self.policies_added = 0
        self.policies_rejected = 0
        self.total_evaluations = 0
        self.behavior_cache = {}
        
        print(f"[Enhanced QD Pool] Initialized {agent_type} pool:")
        print(f"  - Behavior dimensions: {behavior_dims}D")
        print(f"  - Grid resolution: {resolution}^{behavior_dims} = {resolution**behavior_dims:,} cells")
        print(f"  - Max archive size: {max_archive_size}")
    
    def add_policy(self, policy, performance, generation, behavior=None, env=None):
        """Add policy to archive if it's elite for its cell"""
        
        # Compute behavior if not provided
        if behavior is None:
            if env is None:
                raise ValueError("Must provide either 'behavior' or 'env'")
            behavior = self.get_behavior_descriptor(policy, env)
        
        # Validate behavior
        if not self._validate_behavior(behavior):
            self.policies_rejected += 1
            return False
        
        # Discretize to cell
        cell = self._behavior_to_cell(behavior)
        
        # Add if better or new cell
        if cell not in self.archive or performance > self.archive[cell].fitness:
            self.archive[cell] = PolicyCell(policy, performance, behavior, generation)
            self.policies_added += 1
            
            # Prune if needed
            if len(self.archive) > self.max_archive_size:
                self._prune_archive()
            
            return True
        else:
            self.policies_rejected += 1
            return False
    
    def sample_opponent(self, strategy='uniform'):
        """Sample opponent from diverse archive"""
        if not self.archive:
            return None
        
        if strategy == 'uniform':
            cell = random.choice(list(self.archive.keys()))
            return self.archive[cell].policy
        
        elif strategy == 'fitness_proportional':
            cells = list(self.archive.keys())
            fitnesses = np.array([self.archive[cell].fitness for cell in cells])
            exp_fitnesses = np.exp(fitnesses - np.max(fitnesses))
            probs = exp_fitnesses / np.sum(exp_fitnesses)
            idx = np.random.choice(len(cells), p=probs)
            return self.archive[cells[idx]].policy
        
        elif strategy == 'recent':
            cells = list(self.archive.keys())
            generations = [self.archive[cell].generation for cell in cells]
            recent_threshold = max(generations) - 2
            recent_cells = [cell for cell in cells 
                          if self.archive[cell].generation >= recent_threshold]
            if recent_cells:
                cell = random.choice(recent_cells)
                return self.archive[cell].policy
            else:
                cell = random.choice(cells)
                return self.archive[cell].policy
        
        else:
            raise ValueError(f"Unknown sampling strategy: {strategy}")
    
    # ========================================================================

    # INNOVATION #2: Enhanced Behavioral Characterization (3D) - FIXED
>>>>>>> 88d0ae3 (Syncing local files with GitHub)
    # ========================================================================
    
    def get_behavior_descriptor(self, policy, env, num_episodes=5):
        """

        INNOVATION #2 ENHANCED: 3D Behavioral Characterization (AGENT-AWARE)
        
        Behavioral dimensions:
        - Dimension 1: Kill chain progression (Red) / Detection improvement (Blue)
        - Dimension 2: Tool diversity (entropy) - both agents
        - Dimension 3: Effectiveness - damage for Red, healing for Blue
>>>>>>> 88d0ae3 (Syncing local files with GitHub)
        
        Args:
            policy: RL policy to evaluate
            env: ACEACv2Environment
            num_episodes: Number of episodes to average
        
        Returns:
 

            tuple: (dim1, tool_diversity, effectiveness) in [0,1]
        """
        self.total_evaluations += 1
        
        dim1_values = []
        tool_entropies = []
        effectiveness_scores = []
        
        is_blue = (self.agent_type == "blue")
        
>>>>>>> 88d0ae3 (Syncing local files with GitHub)
        for ep in range(num_episodes):
            obs, _ = env.reset()
            tool_usage = defaultdict(int)
            steps = 0
            total_reward = 0
            initial_security = getattr(env, 'network_security', 0.8)

            initial_detection = getattr(env, 'detection_level', 0.3)
>>>>>>> 88d0ae3 (Syncing local files with GitHub)
            
            for step in range(100):
                action, _ = policy.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = env.step(action)
                
                # Track tool usage
                if isinstance(action, np.ndarray):
                    action = int(action.item())
                tool_usage[action] += 1
                total_reward += reward
                steps += 1
                
                if done:
                    break
            
            # Dimension 1: Agent-specific progression
            if is_blue:
                # Blue: Detection improvement
                final_detection = info.get('detection_level', initial_detection)
                detection_gain = final_detection - initial_detection
                dim1 = np.clip(detection_gain / (1.0 - initial_detection + 1e-6), 0.0, 1.0)
            else:
                # Red: Kill chain progression
                kill_chain_progress = info.get('kill_chain_progress', 0.0)
                dim1 = kill_chain_progress
            
            dim1_values.append(dim1)
            
            # Dimension 2: Tool diversity (Shannon entropy) - FIXED
            counts = np.array(list(tool_usage.values()), dtype=float)
            if len(counts) > 1 and counts.sum() > 0:
                probs = counts / counts.sum()
                entropy = -np.sum(probs * np.log(probs + 1e-10))
                max_entropy = np.log(len(counts))  # Max entropy for tools used
                normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
            else:
                normalized_entropy = 0.0
            tool_entropies.append(max(0.0, normalized_entropy))  # Prevent -0.000
            
            # Dimension 3: Agent-specific effectiveness - FIXED FOR BLUE
            final_security = info.get('network_security', initial_security)
            
            if is_blue:
                # Blue: Healing effectiveness (security INCREASE)
                security_gain = final_security - initial_security
                effectiveness = np.clip(security_gain / (1.0 - initial_security + 1e-6), 0.0, 1.0)
            else:
                # Red: Attack effectiveness (security REDUCTION)
                security_reduction = initial_security - final_security
                effectiveness = np.clip(security_reduction / (initial_security + 1e-6), 0.0, 1.0)
            
>>>>>>> 88d0ae3 (Syncing local files with GitHub)
            effectiveness_scores.append(effectiveness)
        
        # Average over episodes
        behavior = (

            float(np.mean(dim1_values)),           # Agent-specific
            float(np.mean(tool_entropies)),        # Tool diversity
            float(np.mean(effectiveness_scores))   # Effectiveness
>>>>>>> 88d0ae3 (Syncing local files with GitHub)
        )
        
        return behavior
    
    # ========================================================================
    # Helper Methods
    # ========================================================================
    
    def _validate_behavior(self, behavior):
        """Validate behavior descriptor"""
        if not isinstance(behavior, (tuple, list)):
            print(f"[QD Pool] Warning: behavior not tuple/list: {type(behavior)}")
            return False
        
        if len(behavior) != self.behavior_dims:
            print(f"[QD Pool] Warning: behavior dims mismatch: {len(behavior)} != {self.behavior_dims}")
            return False
        
        if not all(np.isfinite(b) for b in behavior):
            print(f"[QD Pool] Warning: behavior contains NaN/Inf: {behavior}")
            return False
        
        return True
    
    def _behavior_to_cell(self, behavior):
        """Convert continuous behavior to discrete grid cell"""
        cell = []
        for i, b in enumerate(behavior):
            b_clipped = np.clip(b, 0.0, 0.9999)
            cell_coord = int(b_clipped * self.resolution)
            cell.append(cell_coord)
        return tuple(cell)
    
    def _prune_archive(self):
        """Remove worst policies when archive exceeds max size"""
        if len(self.archive) <= self.max_archive_size:
            return
        
        sorted_cells = sorted(
            self.archive.items(),
            key=lambda x: x[1].fitness
        )
        
        keep_size = int(self.max_archive_size * 0.9)
        self.archive = dict(sorted_cells[-keep_size:])
        
        print(f"[QD Pool] Pruned archive: {len(sorted_cells)} → {len(self.archive)} policies")
    
    def get_stats(self):
        """Get detailed pool statistics"""
        if not self.archive:
            return {
                'size': 0,
                'avg': 0.0,
                'best': 0.0,
                'worst': 0.0,
                'coverage': 0.0,
                'policies_added': self.policies_added,
                'policies_rejected': self.policies_rejected,
                'total_evaluations': self.total_evaluations
            }
        
        fitnesses = [cell.fitness for cell in self.archive.values()]
        total_cells = self.resolution ** self.behavior_dims
        coverage = len(self.archive) / total_cells
        
        return {
            'size': len(self.archive),
            'avg': float(np.mean(fitnesses)),
            'best': float(np.max(fitnesses)),
            'worst': float(np.min(fitnesses)),
            'coverage': float(coverage),
            'policies_added': self.policies_added,
            'policies_rejected': self.policies_rejected,
            'acceptance_rate': self.policies_added / max(self.policies_added + self.policies_rejected, 1),
            'total_evaluations': self.total_evaluations
        }
    
    def save_archive(self, path):
        """Save archive to disk"""
        os.makedirs(path, exist_ok=True)
        
        archive_data = {
            'metadata': {
                'agent_type': self.agent_type,
                'behavior_dims': self.behavior_dims,
                'resolution': self.resolution,
                'max_archive_size': self.max_archive_size,
                'policies_added': self.policies_added,
                'policies_rejected': self.policies_rejected,
                'total_evaluations': self.total_evaluations
            },
            'cells': {}
        }
        
        for cell, policy_cell in self.archive.items():
            cell_str = '_'.join(map(str, cell))
            policy_path = f"{path}/cell_{cell_str}.zip"
            
            policy_cell.policy.save(policy_path)
            
            archive_data['cells'][str(cell)] = {
                'fitness': policy_cell.fitness,
                'behavior': policy_cell.behavior,
                'generation': policy_cell.generation,
                'policy_path': policy_path
            }
        
        with open(f"{path}/archive.json", 'w') as f:
            json.dump(archive_data, f, indent=2)
        
        print(f"[QD Pool] Saved archive: {len(self.archive)} policies to {path}")


# ============================================================================
# SWAP RL Trainer (Enhanced)
# ============================================================================

class EnhancedSWAPRLTrainer:
    """Enhanced SWAP RL Trainer with improved QD"""
    
    def __init__(self, config=None):
        if config is None:
            config = QDConfig()
        
        self.config = config
        
        self.red_pool = EnhancedQualityDiversityPool(
            pool_size=config.POOL_SIZE,
            agent_type="red",
            behavior_dims=config.BEHAVIOR_DIMS,
            resolution=config.GRID_RESOLUTION,
            max_archive_size=config.MAX_ARCHIVE_SIZE
        )
        self.blue_pool = EnhancedQualityDiversityPool(
            pool_size=config.POOL_SIZE,
            agent_type="blue",
            behavior_dims=config.BEHAVIOR_DIMS,
            resolution=config.GRID_RESOLUTION,
            max_archive_size=config.MAX_ARCHIVE_SIZE
        )
        self.history = []
    
    def evaluate_policy(self, policy, env, num_episodes=5):
        """Evaluate policy performance"""
        total_reward = 0.0
        
        for ep in range(num_episodes):
            obs, _ = env.reset()
            ep_reward = 0.0
            
            for step in range(100):
                action, _ = policy.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                ep_reward += reward
                if terminated:
                    break
            
            total_reward += ep_reward
        
        return total_reward / num_episodes
    
    def train_generation(self, red_policy, blue_policy, generation, episodes_per_gen):
        """Train one generation with enhanced QD"""
        
        print("")
        print("="*60)
        print(f"Generation {generation}/{self.config.NUM_GENERATIONS}")
        print("="*60)
        
        gen_start = time.time()
        
        # Sample diverse opponents
        red_opponent = self.blue_pool.sample_opponent(strategy='uniform')
        blue_opponent = self.red_pool.sample_opponent(strategy='uniform')
        
        if red_opponent is None:
            red_opponent = blue_policy
        if blue_opponent is None:
            blue_opponent = red_policy
        
        # Train Red
        print("Training Red Agent...")
        env_red = ACEACv2Environment(agent_role="red")
        red_policy.set_env(env_red)
        
        red_start = time.time()
        red_policy.learn(
            total_timesteps=episodes_per_gen * 100,
            reset_num_timesteps=False,
            progress_bar=True  # Show progress for longer training
        )
        red_duration = time.time() - red_start
        
        # Evaluate Red
        red_performance = self.evaluate_policy(red_policy, env_red, 
                                               num_episodes=self.config.PERFORMANCE_EVAL_EPISODES)
        
        # Compute Red behavior (3D)
        red_behavior = self.red_pool.get_behavior_descriptor(
            red_policy, env_red,
            num_episodes=self.config.BEHAVIOR_EVAL_EPISODES
        )
        red_added = self.red_pool.add_policy(red_policy, red_performance, 
                                            generation, behavior=red_behavior)
        
        print(f"Red: perf={red_performance:.2f}, behavior={tuple(f'{b:.3f}' for b in red_behavior)}, "
              f"added={red_added}, time={red_duration:.1f}s")
        
        # Train Blue
        print("\nTraining Blue Agent...")
        env_blue = ACEACv2Environment(agent_role="blue")
        blue_policy.set_env(env_blue)
        
        blue_start = time.time()
        blue_policy.learn(
            total_timesteps=episodes_per_gen * 100,
            reset_num_timesteps=False,
            progress_bar=True
        )
        blue_duration = time.time() - blue_start
        
        # Evaluate Blue
        blue_performance = self.evaluate_policy(blue_policy, env_blue,
                                               num_episodes=self.config.PERFORMANCE_EVAL_EPISODES)
        
        # Compute Blue behavior (3D)
        blue_behavior = self.blue_pool.get_behavior_descriptor(
            blue_policy, env_blue,
            num_episodes=self.config.BEHAVIOR_EVAL_EPISODES
        )
        blue_added = self.blue_pool.add_policy(blue_policy, blue_performance,
                                              generation, behavior=blue_behavior)
        
        print(f"Blue: perf={blue_performance:.2f}, behavior={tuple(f'{b:.3f}' for b in blue_behavior)}, "
              f"added={blue_added}, time={blue_duration:.1f}s")
        
        gen_duration = time.time() - gen_start
        
        # Get stats
        red_stats = self.red_pool.get_stats()
        blue_stats = self.blue_pool.get_stats()
        
        gen_stats = {
            'generation': generation,
            'duration': gen_duration,
            'red_performance': red_performance,
            'blue_performance': blue_performance,
            'red_behavior': red_behavior,
            'blue_behavior': blue_behavior,
            'red_pool': red_stats,
            'blue_pool': blue_stats
        }
        
        self.history.append(gen_stats)
        
        print(f"\nGen {generation} Summary:")
        print(f"  Duration: {gen_duration:.1f}s")
        print(f"  Red: size={red_stats['size']}, coverage={red_stats['coverage']:.2%}, "
              f"acceptance={red_stats['acceptance_rate']:.1%}")
        print(f"  Blue: size={blue_stats['size']}, coverage={blue_stats['coverage']:.2%}, "
              f"acceptance={blue_stats['acceptance_rate']:.1%}")
        
        return red_policy, blue_policy, gen_stats


# ============================================================================
# Main Training Function (Enhanced)
# ============================================================================

def train_aceac_v2_enhanced(config=None):
    """
    Enhanced QD-SWAP RL Training with ALL IMPROVEMENTS
    
    IMPROVEMENTS:
    A. Grid Resolution: 10x10 → 20x20x20
    B. 3rd Behavioral Dimension: Added effectiveness
    C. Train Longer: 10 → 20 generations
    D. More Episodes: 100 → 200 per generation
    """
    
    if config is None:
        config = QDConfig()
    
    print("")
    print("="*70)
    print("ACEAC v2.0 ENHANCED QD-SWAP RL Training")
    print("="*70)
    print("User: sarowarzahan414")
    print("Date: " + datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"))
    print("")
    print("IMPROVEMENTS:")
    print("  A. Grid Resolution: 20x20x20 (8,000 cells)")
    print("  B. 3rd Behavioral Dim: Effectiveness added")
    print("  C. Longer Training: 20 generations")
    print("  D. More Episodes: 200 per generation")
    print("")
    print(str(config))
    print("="*70)
    
    # Create directories
    os.makedirs("models/aceac_v2_enhanced", exist_ok=True)
    os.makedirs("models/aceac_v2_enhanced/red_pool", exist_ok=True)
    os.makedirs("models/aceac_v2_enhanced/blue_pool", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # Initialize environments
    env_red = ACEACv2Environment(agent_role="red")
    env_blue = ACEACv2Environment(agent_role="blue")
    
    print("")
    print("Initializing agents...")
    red_policy = PPO("MlpPolicy", env_red, verbose=0, 
                    learning_rate=3e-4, n_steps=2048, batch_size=64,
                ent_coef=0.1)
    blue_policy = PPO("MlpPolicy", env_blue, verbose=0,
                     learning_rate=3e-4, n_steps=2048, batch_size=64,
                ent_coef=0.1)
    
    print("Red: 62D obs, 25 offensive tools")
    print("Blue: 62D obs, 25 defensive tools")
    
    trainer = EnhancedSWAPRLTrainer(config)
    
    total_start = time.time()
    
    # Training loop
    for gen in range(1, config.NUM_GENERATIONS + 1):
        red_policy, blue_policy, stats = trainer.train_generation(
            red_policy, blue_policy, gen, config.EPISODES_PER_GEN
        )
        
        # Save snapshots every 5 generations
        if gen % 5 == 0:
            red_policy.save(f"models/aceac_v2_enhanced/red_pool/red_gen{gen}.zip")
            blue_policy.save(f"models/aceac_v2_enhanced/blue_pool/blue_gen{gen}.zip")
            
            # Save archives
            trainer.red_pool.save_archive(f"models/aceac_v2_enhanced/red_archive_gen{gen}")
            trainer.blue_pool.save_archive(f"models/aceac_v2_enhanced/blue_archive_gen{gen}")
    
    total_duration = time.time() - total_start
    
    # Save final models
    red_policy.save("models/aceac_v2_enhanced/red_final.zip")
    blue_policy.save("models/aceac_v2_enhanced/blue_final.zip")
    
    # Save final archives
    trainer.red_pool.save_archive("models/aceac_v2_enhanced/red_archive_final")
    trainer.blue_pool.save_archive("models/aceac_v2_enhanced/blue_archive_final")
    
    # Save training log
    log_data = {
        'user': 'sarowarzahan414',
        'date': datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        'time_utc': datetime.now(timezone.utc).strftime("%H:%M:%S"),
        'improvements': {
            'A': f'Grid resolution: {config.GRID_RESOLUTION}^{config.BEHAVIOR_DIMS}',
            'B': f'Behavioral dims: {config.BEHAVIOR_DIMS}D (added effectiveness)',
            'C': f'Generations: {config.NUM_GENERATIONS}',
            'D': f'Episodes/gen: {config.EPISODES_PER_GEN}'
        },
        'config': {
            'grid_resolution': config.GRID_RESOLUTION,
            'behavior_dims': config.BEHAVIOR_DIMS,
            'num_generations': config.NUM_GENERATIONS,
            'episodes_per_gen': config.EPISODES_PER_GEN,
            'pool_size': config.POOL_SIZE,
            'max_archive_size': config.MAX_ARCHIVE_SIZE
        },
        'duration_seconds': total_duration,
        'duration_minutes': total_duration / 60,
        'duration_hours': total_duration / 3600,
        'history': trainer.history,
        'red_pool_final': trainer.red_pool.get_stats(),
        'blue_pool_final': trainer.blue_pool.get_stats(),
        'status': 'complete'
    }
    
    log_path = 'logs/aceac_v2_enhanced.json'
    with open(log_path, 'w') as f:
        json.dump(log_data, f, indent=2)
    
    # Print final summary
    print("")
    print("="*70)
    print("ENHANCED TRAINING COMPLETE!")
    print("="*70)
    print(f"Total Duration: {total_duration/3600:.2f} hours ({total_duration/60:.1f} minutes)")
    print(f"Total Episodes: {config.NUM_GENERATIONS * config.EPISODES_PER_GEN * 2:,}")
    print("")
    print("Final Results:")
    red_final = trainer.red_pool.get_stats()
    blue_final = trainer.blue_pool.get_stats()
    print(f"  Red Archive:")
    print(f"    Size: {red_final['size']} policies")
    print(f"    Coverage: {red_final['coverage']:.2%} ({red_final['size']}/{config.GRID_RESOLUTION**config.BEHAVIOR_DIMS})")
    print(f"    Best Fitness: {red_final['best']:.2f}")
    print(f"    Acceptance Rate: {red_final['acceptance_rate']:.1%}")
    print(f"  Blue Archive:")
    print(f"    Size: {blue_final['size']} policies")
    print(f"    Coverage: {blue_final['coverage']:.2%} ({blue_final['size']}/{config.GRID_RESOLUTION**config.BEHAVIOR_DIMS})")
    print(f"    Best Fitness: {blue_final['best']:.2f}")
    print(f"    Acceptance Rate: {blue_final['acceptance_rate']:.1%}")
    print("")
    print("Models:")
    print("  Red: models/aceac_v2_enhanced/red_final.zip")
    print("  Blue: models/aceac_v2_enhanced/blue_final.zip")
    print(f"Log: {log_path}")
    print("="*70)
    
    return red_policy, blue_policy, trainer


if __name__ == "__main__":
    # Create custom config or use defaults
    config = QDConfig()
    
    print("\n🚀 Starting Enhanced QD-SWAP RL Training...")
    print(f"Expected duration: ~{config.NUM_GENERATIONS * config.EPISODES_PER_GEN * 2 * 0.5 / 60:.1f} hours")
    print("(Estimate: ~0.5 min per 200 episodes)\n")
    
    red, blue, trainer = train_aceac_v2_enhanced(config)
    
    print("\n✅ Enhanced ACEAC v2.0 QD-SWAP RL Complete!")
    print(f"Check logs/aceac_v2_enhanced.json for detailed results")
