"""
ACEAC Blue Agent - 100 Episode Validation (FIXED)
Quick test before scaling to 10k episodes

Author: @sarowarzahan414
Date: 2025-10-08 21:46 UTC
Location: Kali Linux VirtualBox
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import PPO
import json
import time
from datetime import datetime, timezone
import os

class ACEACCyberRangeBlue(gym.Env):
    """ACEAC Blue Agent - Defensive AI (FIXED)"""
    
    metadata = {'render_modes': [], 'name': 'ACEAC-Blue-v1'}
    
    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(20,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(10)  # 10 defensive actions
        self.max_steps = 50
        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.defense_level = 0.5
        self.attacks_blocked = 0
        self.total_attacks = 0
        return self._get_observation(), {}
    
    def step(self, action):
        self.current_step += 1
        self.total_attacks += 1
        
        # Convert action to int if it's an array (FIX!)
        if isinstance(action, np.ndarray):
            action = int(action.item())
        else:
            action = int(action)
        
        # Simulate attack
        attack_strength = np.random.uniform(0.3, 0.9)
        
        # Defense boost from action
        defense_boost = self._get_defense_boost(action)
        self.defense_level = np.clip(
            self.defense_level + defense_boost, 0.0, 1.0
        )
        
        # Reward calculation
        if self.defense_level > attack_strength:
            reward = 10.0 + (self.defense_level - attack_strength) * 5.0
            self.attacks_blocked += 1
        else:
            reward = -5.0 - (attack_strength - self.defense_level) * 10.0
        
        # Action costs
        if action in [1, 2, 5]:
            reward -= 1.0
        
        terminated = self.current_step >= self.max_steps
        
        info = {
            'defense_level': float(self.defense_level),
            'attacks_blocked': int(self.attacks_blocked),
            'block_rate': float(self.attacks_blocked / self.total_attacks)
        }
        
        return self._get_observation(), float(reward), terminated, False, info
    
    def _get_defense_boost(self, action):
        """Get defense boost - action is guaranteed to be int"""
        boosts = {
            0: 0.02,  # Monitor traffic
            1: 0.08,  # Block IP
            2: 0.10,  # Update firewall
            3: 0.06,  # Patch system
            4: 0.07,  # Enable IDS
            5: 0.09,  # Isolate host
            6: 0.03,  # Analyze logs
            7: 0.05,  # Strengthen auth
            8: 0.04,  # Backup data
            9: 0.01   # Alert admin
        }
        return float(boosts[action] + np.random.uniform(-0.01, 0.01))
    
    def _get_observation(self):
        obs = np.zeros(20, dtype=np.float32)
        obs[0] = self.defense_level
        obs[1] = self.attacks_blocked / (self.total_attacks + 1)
        obs[2] = self.current_step / self.max_steps
        obs[3:] = np.random.random(17) * 0.5
        return obs


def train_blue_agent_100ep():
    """Train Blue Agent - 100 episodes for validation"""
    
    print("\n" + "="*70)
    print("🛡️  ACEAC BLUE AGENT - 100 EPISODE VALIDATION (FIXED)")
    print("="*70)
    print(f"User: sarowarzahan414")
    print(f"Date: 2025-10-08")
    print(f"Time: 21:46:00 UTC")
    print(f"Location: Kali Linux VirtualBox (Local)")
    print(f"Purpose: Validate before scaling to 10k")
    print("="*70 + "\n")
    
    # Create directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # Create environment
    env = ACEACCyberRangeBlue()
    
    print("🤖 Creating Blue Agent (Defensive AI - PPO)...")
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
    )
    
    # Training parameters
    episodes = 100
    timesteps = episodes * 50  # 5,000 timesteps
    
    print(f"\n📊 Configuration:")
    print(f"   Episodes: {episodes}")
    print(f"   Timesteps: {timesteps:,}")
    print(f"   Algorithm: PPO (Proximal Policy Optimization)")
    print(f"   Defensive Actions: 10")
    print()
    
    # Train
    print("🔄 Training Blue Agent...\n")
    start_time = time.time()
    
    model.learn(total_timesteps=timesteps, progress_bar=True)
    
    end_time = time.time()
    duration = end_time - start_time
    
    # Save model
    model_path = "models/aceac_blue_agent_100ep.zip"
    model.save(model_path)
    
    print("\n🧪 Testing trained Blue Agent...")
    
    # Test the trained agent
    episode_rewards = []
    block_rates = []
    
    for ep in range(10):
        ep_reward = 0
        obs, _ = env.reset()
        
        for step in range(50):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            
            if terminated:
                break
        
        episode_rewards.append(float(ep_reward))
        block_rates.append(info['block_rate'])
        print(f"  Test Episode {ep+1}: Reward={ep_reward:.1f}, Block Rate={info['block_rate']:.1%}")
    
    avg_reward = sum(episode_rewards) / len(episode_rewards)
    avg_block_rate = sum(block_rates) / len(block_rates)
    
    # Save log
    log_data = {
        "agent_type": "blue_defensive",
        "episodes": episodes,
        "total_timesteps": timesteps,
        "algorithm": "PPO",
        "training_location": "kali_linux_virtualbox",
        "user": "sarowarzahan414",
        "date": "2025-10-08",
        "time_utc": "21:46:00",
        "start_time": datetime.fromtimestamp(start_time, timezone.utc).isoformat(),
        "end_time": datetime.fromtimestamp(end_time, timezone.utc).isoformat(),
        "duration_seconds": duration,
        "timesteps_per_second": timesteps / duration,
        "model_path": model_path,
        "test_episodes": 10,
        "test_rewards": episode_rewards,
        "avg_test_reward": avg_reward,
        "avg_block_rate": avg_block_rate,
        "status": "validation_complete",
        "bug_fixed": "numpy_array_to_int_conversion"
    }
    
    log_path = "logs/aceac_blue_100ep_validation.json"
    with open(log_path, "w") as f:
        json.dump(log_data, f, indent=2)
    
    # Results
    print("\n" + "="*70)
    print("✅ VALIDATION TRAINING COMPLETE!")
    print("="*70)
    print(f"Duration: {duration:.2f} seconds")
    print(f"Speed: {timesteps/duration:.0f} timesteps/second")
    print(f"Avg Test Reward: {avg_reward:.2f} (10 test episodes)")
    print(f"Avg Block Rate: {avg_block_rate:.1%}")
    print(f"Model: {model_path}")
    print(f"Log: {log_path}")
    print("="*70)
    print("\n🎯 Ready for Red vs Blue validation!")
    print("="*70 + "\n")
    
    return model, log_data


if __name__ == "__main__":
    print("🚀 Starting ACEAC Blue Agent Training (FIXED VERSION)...")
    model, log = train_blue_agent_100ep()
    print("🎉 Blue Agent validation complete!")
    print("📋 Next: Validate Red vs Blue interaction")
