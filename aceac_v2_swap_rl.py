"""
ACEAC v2.0 - SWAP RL Training (FINAL)
Self-Play with Adaptive Policies + Cyber Kill Chain

Author: @sarowarzahan414
Date: 2025-10-08 23:28:08 UTC
Location: Kali Linux VirtualBox
"""

import numpy as np
from stable_baselines3 import PPO
from aceac_v2_cyber_killchain import ACEACv2Environment
import json
import time
from datetime import datetime, timezone
import os
import pickle


class PolicyPool:
    """Policy pool for SWAP RL"""
    
    def __init__(self, pool_size=5, agent_type="red"):
        self.pool_size = pool_size
        self.agent_type = agent_type
        self.policies = []
        self.policy_performances = []
        self.policy_generations = []
        
    def add_policy(self, policy, performance, generation):
        """Add policy to pool"""
        self.policies.append(policy)
        self.policy_performances.append(performance)
        self.policy_generations.append(generation)
        
        if len(self.policies) > self.pool_size:
            worst_idx = np.argmin(self.policy_performances)
            self.policies.pop(worst_idx)
            self.policy_performances.pop(worst_idx)
            self.policy_generations.pop(worst_idx)
    
    def sample_opponent(self):
        """Sample random opponent from pool"""
        if not self.policies:
            return None
        idx = np.random.randint(0, len(self.policies))
        return self.policies[idx]
    
    def get_stats(self):
        """Get pool statistics"""
        if not self.policy_performances:
            return {'size': 0, 'avg': 0.0, 'best': 0.0, 'worst': 0.0}
        
        return {
            'size': len(self.policies),
            'avg': float(np.mean(self.policy_performances)),
            'best': float(np.max(self.policy_performances)),
            'worst': float(np.min(self.policy_performances))
        }


class SWAPRLTrainer:
    """SWAP RL Trainer"""
    
    def __init__(self, red_pool_size=5, blue_pool_size=5):
        self.red_pool = PolicyPool(pool_size=red_pool_size, agent_type="red")
        self.blue_pool = PolicyPool(pool_size=blue_pool_size, agent_type="blue")
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
    
    def train_generation(self, red_policy, blue_policy, generation, episodes_per_gen=100):
        """Train one generation"""
        
        print("")
        print("="*50)
        print("Generation " + str(generation))
        print("="*50)
        
        gen_start = time.time()
        
        # Sample opponents
        red_opponent = self.blue_pool.sample_opponent()
        blue_opponent = self.red_pool.sample_opponent()
        
        if red_opponent is None:
            red_opponent = blue_policy
        if blue_opponent is None:
            blue_opponent = red_policy
        
        # Train Red
        print("Training Red Agent...")
        env_red = ACEACv2Environment(agent_role="red")
        red_policy.set_env(env_red)
        
        red_start = time.time()
        red_policy.learn(total_timesteps=episodes_per_gen * 100, reset_num_timesteps=False, progress_bar=False)
        red_duration = time.time() - red_start
        
        red_performance = self.evaluate_policy(red_policy, env_red)
        print("Red: " + str(round(red_performance, 2)) + " (" + str(round(red_duration, 1)) + "s)")
        
        self.red_pool.add_policy(red_policy, red_performance, generation)
        
        # Train Blue
        print("Training Blue Agent...")
        env_blue = ACEACv2Environment(agent_role="blue")
        blue_policy.set_env(env_blue)
        
        blue_start = time.time()
        blue_policy.learn(total_timesteps=episodes_per_gen * 100, reset_num_timesteps=False, progress_bar=False)
        blue_duration = time.time() - blue_start
        
        blue_performance = self.evaluate_policy(blue_policy, env_blue)
        print("Blue: " + str(round(blue_performance, 2)) + " (" + str(round(blue_duration, 1)) + "s)")
        
        self.blue_pool.add_policy(blue_policy, blue_performance, generation)
        
        gen_duration = time.time() - gen_start
        
        gen_stats = {
            'generation': generation,
            'duration': gen_duration,
            'red_performance': red_performance,
            'blue_performance': blue_performance,
            'red_pool': self.red_pool.get_stats(),
            'blue_pool': self.blue_pool.get_stats()
        }
        
        self.history.append(gen_stats)
        
        print("Gen " + str(generation) + " done (" + str(round(gen_duration, 1)) + "s)")
        print("Red pool: " + str(self.red_pool.get_stats()['size']) + " | Blue pool: " + str(self.blue_pool.get_stats()['size']))
        
        return red_policy, blue_policy, gen_stats


def train_aceac_v2(episodes_per_gen=100, num_gens=10, pool_size=5):
    """Main training function"""
    
    print("")
    print("="*70)
    print("ACEAC v2.0 SWAP RL Training")
    print("="*70)
    print("User: sarowarzahan414")
    print("Date: 2025-10-08 23:28:08 UTC")
    print("Cyber Kill Chain + 50 Real Tools + SWAP RL")
    print("-"*70)
    print("Generations: " + str(num_gens))
    print("Episodes/gen: " + str(episodes_per_gen))
    print("Pool size: " + str(pool_size))
    print("Total episodes: " + str(episodes_per_gen * num_gens * 2))
    print("="*70)
    
    os.makedirs("models/aceac_v2", exist_ok=True)
    os.makedirs("models/aceac_v2/red_pool", exist_ok=True)
    os.makedirs("models/aceac_v2/blue_pool", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    env_red = ACEACv2Environment(agent_role="red")
    env_blue = ACEACv2Environment(agent_role="blue")
    
    print("")
    print("Initializing agents...")
    red_policy = PPO("MlpPolicy", env_red, verbose=0, learning_rate=3e-4, n_steps=2048, batch_size=64)
    blue_policy = PPO("MlpPolicy", env_blue, verbose=0, learning_rate=3e-4, n_steps=2048, batch_size=64)
    
    print("Red: 62D obs, 25 offensive tools")
    print("Blue: 62D obs, 25 defensive tools")
    
    trainer = SWAPRLTrainer(red_pool_size=pool_size, blue_pool_size=pool_size)
    
    total_start = time.time()
    
    for gen in range(1, num_gens + 1):
        red_policy, blue_policy, stats = trainer.train_generation(red_policy, blue_policy, gen, episodes_per_gen)
        
        red_policy.save("models/aceac_v2/red_pool/red_gen" + str(gen) + ".zip")
        blue_policy.save("models/aceac_v2/blue_pool/blue_gen" + str(gen) + ".zip")
    
    total_duration = time.time() - total_start
    
    red_policy.save("models/aceac_v2/red_final.zip")
    blue_policy.save("models/aceac_v2/blue_final.zip")
    
    log_data = {
        'user': 'sarowarzahan414',
        'date': '2025-10-08',
        'time_utc': '23:28:08',
        'duration_seconds': total_duration,
        'duration_minutes': total_duration / 60,
        'generations': num_gens,
        'episodes_per_gen': episodes_per_gen,
        'pool_size': pool_size,
        'history': trainer.history,
        'red_pool_final': trainer.red_pool.get_stats(),
        'blue_pool_final': trainer.blue_pool.get_stats(),
        'status': 'complete'
    }
    
    log_path = 'logs/aceac_v2_swap_rl.json'
    with open(log_path, 'w') as f:
        json.dump(log_data, f, indent=2)
    
    print("")
    print("="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print("Duration: " + str(round(total_duration/60, 1)) + " minutes")
    print("Episodes: " + str(episodes_per_gen * num_gens * 2))
    print("")
    print("Models:")
    print("  Red: models/aceac_v2/red_final.zip")
    print("  Blue: models/aceac_v2/blue_final.zip")
    print("")
    print("Pools:")
    print("  Red: " + str(trainer.red_pool.get_stats()['size']) + " policies")
    print("  Blue: " + str(trainer.blue_pool.get_stats()['size']) + " policies")
    print("")
    print("Log: " + log_path)
    print("="*70)
    print("")
    
    return red_policy, blue_policy, trainer


if __name__ == "__main__":
    red, blue, trainer = train_aceac_v2(episodes_per_gen=100, num_gens=10, pool_size=5)
    print("ACEAC v2.0 SWAP RL Complete!")
