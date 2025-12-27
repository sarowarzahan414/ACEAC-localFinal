"""
ACEAC v2.0 - QD-SWAP RL Training (COMPLETE SOLUTION)
Quality-Diversity + Self-Play with Adaptive Policies + Superior Behavioral Dimensions

Author: @sarowarzahan414
Date: 2025-12-27
Location: Claude Code Implementation

Features:
- MAP-Elites archive for Quality-Diversity
- 3D behavioral characterization (20x20x20 = 8,000 cells)
- Superior Blue behavioral dimensions: detection_rate, response_rate, efficiency
- Dense reward shaping for diverse defensive strategies
- Strong exploration (ent_coef=0.5)
"""

import numpy as np
from stable_baselines3 import PPO
from aceac_v2_cyber_killchain import ACEACv2Environment
import json
import time
from datetime import datetime
import os
import pickle


class QDArchive:
    """MAP-Elites Archive for Quality-Diversity"""

    def __init__(self, grid_resolution=20, num_dimensions=3, agent_type="red"):
        self.grid_resolution = grid_resolution
        self.num_dimensions = num_dimensions
        self.agent_type = agent_type

        # Archive: grid_resolution^num_dimensions cells
        self.archive = {}  # Key: (grid_x, grid_y, grid_z), Value: (policy, performance, behavior)

        self.archive_history = []

    def discretize_behavior(self, behavior):
        """Convert continuous behavior to grid coordinates"""
        coords = []
        for b in behavior:
            # Clip to [0, 1] and discretize
            b_clipped = np.clip(b, 0.0, 1.0)
            grid_idx = int(b_clipped * (self.grid_resolution - 1))
            coords.append(grid_idx)
        return tuple(coords)

    def add_policy(self, policy, performance, behavior, generation):
        """Add policy to archive (MAP-Elites update)"""
        grid_coords = self.discretize_behavior(behavior)

        # If cell is empty OR new policy is better, add it
        if grid_coords not in self.archive or performance > self.archive[grid_coords][1]:
            self.archive[grid_coords] = (policy, performance, behavior, generation)
            return True  # Policy was added
        return False  # Policy was not added (worse than existing)

    def sample_opponent(self):
        """Sample random opponent from archive"""
        if not self.archive:
            return None
        coords = list(self.archive.keys())
        random_coords = coords[np.random.randint(0, len(coords))]
        return self.archive[random_coords][0]  # Return policy

    def get_stats(self):
        """Get archive statistics"""
        if not self.archive:
            return {
                'size': 0,
                'coverage': 0.0,
                'avg_performance': 0.0,
                'best_performance': 0.0,
                'worst_performance': 0.0
            }

        performances = [v[1] for v in self.archive.values()]
        total_cells = self.grid_resolution ** self.num_dimensions

        return {
            'size': len(self.archive),
            'coverage': len(self.archive) / total_cells * 100.0,
            'avg_performance': float(np.mean(performances)),
            'best_performance': float(np.max(performances)),
            'worst_performance': float(np.min(performances))
        }


class QDSWAPTrainer:
    """Quality-Diversity SWAP RL Trainer with Superior Behavioral Dimensions"""

    def __init__(self, grid_resolution=20):
        self.red_archive = QDArchive(grid_resolution=grid_resolution, num_dimensions=3, agent_type="red")
        self.blue_archive = QDArchive(grid_resolution=grid_resolution, num_dimensions=3, agent_type="blue")
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

    def get_red_behavior(self, policy, env, num_episodes=5):
        """Get Red agent behavioral characterization

        Dimensions:
        1. Aggression (network security decrease)
        2. Tool Diversity (entropy of tool usage)
        3. Kill Chain Progress (how far through kill chain)
        """
        aggression_vals = []
        diversity_vals = []
        progress_vals = []

        for ep in range(num_episodes):
            obs, info = env.reset()
            initial_security = env.network_security
            actions_taken = []

            for step in range(100):
                action, _ = policy.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                actions_taken.append(action)
                if terminated:
                    break

            # Dimension 1: Aggression (security decrease)
            security_drop = initial_security - env.network_security
            aggression = np.clip(security_drop / 0.8, 0.0, 1.0)
            aggression_vals.append(aggression)

            # Dimension 2: Tool Diversity (Shannon entropy)
            if len(actions_taken) > 0:
                unique, counts = np.unique(actions_taken, return_counts=True)
                probs = counts / len(actions_taken)
                entropy = -np.sum(probs * np.log(probs + 1e-10))
                max_entropy = np.log(25)  # 25 tools
                diversity = np.clip(entropy / max_entropy, 0.0, 1.0)
            else:
                diversity = 0.0
            diversity_vals.append(diversity)

            # Dimension 3: Kill Chain Progress
            if 'kill_chain_progress' in info:
                progress = np.clip(info['kill_chain_progress'], 0.0, 1.0)
            else:
                progress = 0.0
            progress_vals.append(progress)

        behavior = (
            float(np.mean(aggression_vals)),
            float(np.mean(diversity_vals)),
            float(np.mean(progress_vals))
        )

        return behavior

    def get_blue_behavior_superior(self, policy, env, num_episodes=5):
        """USER'S SUPERIOR APPROACH for Blue behavioral characterization

        Dimensions:
        1. Detection Rate (attacks_detected / total_attacks)
        2. Tool Switching Rate (adaptability - how often Blue changes tools)
        3. Resource Efficiency (reward / tools_used)

        Note: Originally dimension 2 was "response_rate = defensive_actions / timesteps"
        but this is always 1.0 because RL agents must act every timestep.
        Replaced with Tool Switching Rate to measure strategic adaptability.
        """
        detection_rates = []
        switching_rates = []
        efficiencies = []

        for ep in range(num_episodes):
            obs, _ = env.reset()

            # Reset environment tracking
            env.attacks_detected = 0
            env.total_attacks = 0
            env.false_positives = 0
            env.defensive_actions = 0

            total_reward = 0.0
            actions_taken = []
            total_timesteps = 0

            for step in range(100):
                action, _ = policy.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                actions_taken.append(action)
                total_timesteps += 1
                if terminated:
                    break

            # Dimension 1: Detection Rate (attacks caught / total attacks)
            if env.total_attacks > 0:
                detection_rate = env.attacks_detected / env.total_attacks
            else:
                detection_rate = 0.0
            detection_rates.append(detection_rate)

            # Dimension 2: Tool Switching Rate (strategic adaptability)
            # Count how often the tool changes from one step to the next
            if len(actions_taken) > 1:
                switches = sum(1 for i in range(1, len(actions_taken)) if actions_taken[i] != actions_taken[i-1])
                switching_rate = switches / (len(actions_taken) - 1)
            else:
                switching_rate = 0.0
            switching_rates.append(switching_rate)

            # Dimension 3: Resource Efficiency (reward per tool)
            unique_tools = len(set(actions_taken))
            if unique_tools > 0:
                efficiency = total_reward / unique_tools
                efficiency_normalized = np.clip(efficiency / 50.0, 0.0, 1.0)  # Normalize to [0,1]
            else:
                efficiency_normalized = 0.0
            efficiencies.append(efficiency_normalized)

        behavior = (
            float(np.mean(detection_rates)),
            float(np.mean(switching_rates)),
            float(np.mean(efficiencies))
        )

        return behavior

    def train_generation(self, red_policy, blue_policy, generation, episodes_per_gen=200):
        """Train one generation with QD"""

        print("")
        print("=" * 60)
        print(f"Generation {generation}")
        print("=" * 60)

        gen_start = time.time()

        # Sample opponents from archives
        red_opponent = self.blue_archive.sample_opponent()
        blue_opponent = self.red_archive.sample_opponent()

        if red_opponent is None:
            red_opponent = blue_policy
        if blue_opponent is None:
            blue_opponent = red_policy

        # Train Red Agent
        print("Training Red Agent...")
        env_red = ACEACv2Environment(agent_role="red")
        red_policy.set_env(env_red)

        red_start = time.time()
        red_policy.learn(total_timesteps=episodes_per_gen * 100, reset_num_timesteps=False, progress_bar=False)
        red_duration = time.time() - red_start

        red_performance = self.evaluate_policy(red_policy, env_red)
        red_behavior = self.get_red_behavior(red_policy, env_red)

        red_added = self.red_archive.add_policy(red_policy, red_performance, red_behavior, generation)

        print(f"Red: perf={red_performance:.2f}, behavior={tuple(f'{b:.3f}' for b in red_behavior)}, added={red_added} ({red_duration:.1f}s)")

        # Train Blue Agent
        print("Training Blue Agent...")
        env_blue = ACEACv2Environment(agent_role="blue")
        blue_policy.set_env(env_blue)

        blue_start = time.time()
        blue_policy.learn(total_timesteps=episodes_per_gen * 100, reset_num_timesteps=False, progress_bar=False)
        blue_duration = time.time() - blue_start

        blue_performance = self.evaluate_policy(blue_policy, env_blue)
        blue_behavior = self.get_blue_behavior_superior(blue_policy, env_blue)

        blue_added = self.blue_archive.add_policy(blue_policy, blue_performance, blue_behavior, generation)

        print(f"Blue: perf={blue_performance:.2f}, behavior={tuple(f'{b:.3f}' for b in blue_behavior)}, added={blue_added} ({blue_duration:.1f}s)")

        gen_duration = time.time() - gen_start

        # Archive stats
        red_stats = self.red_archive.get_stats()
        blue_stats = self.blue_archive.get_stats()

        gen_stats = {
            'generation': generation,
            'duration': gen_duration,
            'red_performance': red_performance,
            'blue_performance': blue_performance,
            'red_behavior': red_behavior,
            'blue_behavior': blue_behavior,
            'red_added': red_added,
            'blue_added': blue_added,
            'red_archive': red_stats,
            'blue_archive': blue_stats
        }

        self.history.append(gen_stats)

        print(f"Gen {generation} done ({gen_duration:.1f}s)")
        print(f"Red Archive: {red_stats['size']} policies ({red_stats['coverage']:.2f}%)")
        print(f"Blue Archive: {blue_stats['size']} policies ({blue_stats['coverage']:.2f}%)")

        return red_policy, blue_policy, gen_stats


def train_aceac_v2_qd(episodes_per_gen=200, num_gens=20, grid_resolution=20):
    """Main QD-SWAP RL training function"""

    print("")
    print("=" * 70)
    print("ACEAC v2.0 QD-SWAP RL Training (COMPLETE SOLUTION)")
    print("=" * 70)
    print("User: sarowarzahan414")
    print("Date: 2025-12-27")
    print("Innovations: QD MAP-Elites + Superior Blue Behavioral Dimensions")
    print("-" * 70)
    print(f"Generations: {num_gens}")
    print(f"Episodes/gen: {episodes_per_gen}")
    print(f"Grid resolution: {grid_resolution}x{grid_resolution}x{grid_resolution} = {grid_resolution**3} cells")
    print(f"Total timesteps: {episodes_per_gen * 100 * num_gens * 2}")
    print("=" * 70)

    os.makedirs("models/aceac_v2_qd", exist_ok=True)
    os.makedirs("models/aceac_v2_qd/red_archive", exist_ok=True)
    os.makedirs("models/aceac_v2_qd/blue_archive", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    env_red = ACEACv2Environment(agent_role="red")
    env_blue = ACEACv2Environment(agent_role="blue")

    print("")
    print("Initializing agents with STRONG EXPLORATION...")
    # STRONG EXPLORATION: ent_coef=0.5 (default is 0.0)
    red_policy = PPO("MlpPolicy", env_red, verbose=0, learning_rate=3e-4, n_steps=2048, batch_size=64, ent_coef=0.5)
    blue_policy = PPO("MlpPolicy", env_blue, verbose=0, learning_rate=3e-4, n_steps=2048, batch_size=64, ent_coef=0.5)

    print("Red: 62D obs, 25 offensive tools, ent_coef=0.5")
    print("Blue: 62D obs, 25 defensive tools, ent_coef=0.5")
    print("Blue behavioral dimensions: detection_rate, tool_switching_rate, efficiency")

    trainer = QDSWAPTrainer(grid_resolution=grid_resolution)

    total_start = time.time()

    for gen in range(1, num_gens + 1):
        red_policy, blue_policy, stats = trainer.train_generation(red_policy, blue_policy, gen, episodes_per_gen)

        # Save checkpoints every 5 generations
        if gen % 5 == 0:
            red_policy.save(f"models/aceac_v2_qd/red_archive/red_gen{gen}.zip")
            blue_policy.save(f"models/aceac_v2_qd/blue_archive/blue_gen{gen}.zip")

    total_duration = time.time() - total_start

    # Save final policies
    red_policy.save("models/aceac_v2_qd/red_final.zip")
    blue_policy.save("models/aceac_v2_qd/blue_final.zip")

    # Save archives
    with open("models/aceac_v2_qd/red_archive.pkl", "wb") as f:
        pickle.dump(trainer.red_archive.archive, f)
    with open("models/aceac_v2_qd/blue_archive.pkl", "wb") as f:
        pickle.dump(trainer.blue_archive.archive, f)

    log_data = {
        'user': 'sarowarzahan414',
        'date': str(datetime.now().date()),
        'duration_seconds': total_duration,
        'duration_minutes': total_duration / 60,
        'generations': num_gens,
        'episodes_per_gen': episodes_per_gen,
        'grid_resolution': grid_resolution,
        'total_cells': grid_resolution ** 3,
        'history': trainer.history,
        'red_archive_final': trainer.red_archive.get_stats(),
        'blue_archive_final': trainer.blue_archive.get_stats(),
        'status': 'complete'
    }

    log_path = 'logs/aceac_v2_qd_swap_rl.json'
    with open(log_path, 'w') as f:
        json.dump(log_data, f, indent=2)

    print("")
    print("=" * 70)
    print("QD-SWAP RL TRAINING COMPLETE!")
    print("=" * 70)
    print(f"Duration: {total_duration / 60:.1f} minutes")
    print(f"Total timesteps: {episodes_per_gen * 100 * num_gens * 2}")
    print("")
    print("Models:")
    print("  Red: models/aceac_v2_qd/red_final.zip")
    print("  Blue: models/aceac_v2_qd/blue_final.zip")
    print("")
    print("Archives:")
    print(f"  Red: {trainer.red_archive.get_stats()['size']} policies ({trainer.red_archive.get_stats()['coverage']:.2f}% coverage)")
    print(f"  Blue: {trainer.blue_archive.get_stats()['size']} policies ({trainer.blue_archive.get_stats()['coverage']:.2f}% coverage)")
    print("")
    print(f"Log: {log_path}")
    print("=" * 70)
    print("")

    return red_policy, blue_policy, trainer


if __name__ == "__main__":
    red, blue, trainer = train_aceac_v2_qd(episodes_per_gen=200, num_gens=20, grid_resolution=20)
    print("ACEAC v2.0 QD-SWAP RL Complete!")
