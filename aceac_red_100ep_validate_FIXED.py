"""
ACEAC Red Agent - 100 Episode Validation (FIXED)
Offensive AI baseline training for co-evolution

Author: @sarowarzahan414
Date: 2025-10-08
Time: 22:05:30 UTC
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


class ACEACCyberRangeRed(gym.Env):
    """ACEAC Red Agent - Offensive AI (FIXED)"""
    
    metadata = {'render_modes': [], 'name': 'ACEAC-Red-v1'}
    
    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(20,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(10)  # 10 offensive actions
        self.max_steps = 50
        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.network_security = 0.8  # Target starts at 80% security
        self.successful_attacks = 0
        self.total_attempts = 0
        return self._get_observation(), {}
    
    def step(self, action):
        self.current_step += 1
        self.total_attempts += 1
        
        # FIX: Convert numpy array to int
        if isinstance(action, np.ndarray):
            action = int(action.item())
        else:
            action = int(action)
        
        # Get attack power for this action
        attack_power = self._get_attack_power(action)
        
        # Calculate success probability based on current security
        success_prob = attack_power * (1.2 - self.network_security)
        
        # Determine if attack succeeds
        if np.random.random() < success_prob:
            # Attack successful!
            reward = 15.0 + attack_power * 20.0
            self.network_security = max(0.0, self.network_security - attack_power)
            self.successful_attacks += 1
        else:
            # Attack failed
            reward = -8.0
        
        # Stealth penalty for noisy attacks
        if action in [2, 3, 8]:  # Exploit, Privilege escalation, Ransomware
            reward -= 2.0  # Detection penalty
        
        terminated = self.current_step >= self.max_steps
        
        info = {
            'network_security': float(self.network_security),
            'successful_attacks': int(self.successful_attacks),
            'success_rate': float(self.successful_attacks / self.total_attempts)
        }
        
        return self._get_observation(), float(reward), terminated, False, info
    
    def _get_attack_power(self, action):
        """Get attack power for each offensive action"""
        powers = {
            0: 0.08,  # Network scan (reconnaissance)
            1: 0.12,  # Vulnerability probe
            2: 0.18,  # Exploit vulnerability (loud)
            3: 0.22,  # Privilege escalation (loud)
            4: 0.15,  # Lateral movement
            5: 0.16,  # Data exfiltration
            6: 0.14,  # Install persistence
            7: 0.10,  # C2 communication setup
            8: 0.20,  # Deploy ransomware (loud)
            9: 0.09   # Cover tracks
        }
        return float(powers[action] + np.random.uniform(-0.02, 0.02))
    
    def _get_observation(self):
        obs = np.zeros(20, dtype=np.float32)
        obs[0] = self.network_security
        obs[1] = self.successful_attacks / (self.total_attempts + 1)
        obs[2] = self.current_step / self.max_steps
        obs[3:] = np.random.random(17) * 0.5
        return obs


def train_red_agent_100ep():
    """Train Red Agent - 100 episodes validation"""
    
    print("\n" + "="*70)
    print("🔴 ACEAC RED AGENT - 100 EPISODE VALIDATION (FIXED)")
    print("="*70)
    print(f"User: sarowarzahan414")
    print(f"Date: 2025-10-08")
    print(f"Time: 22:05:30 UTC")
    print(f"Location: Kali Linux VirtualBox")
    print(f"Purpose: Baseline validation before co-evolution")
    print("="*70 + "\n")
    
    # Create directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # Create environment
    env = ACEACCyberRangeRed()
    
    print("🤖 Creating Red Agent (Offensive AI - PPO)...")
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
    )
    
    episodes = 100
    timesteps = episodes * 50  # 5,000 timesteps
    
    print(f"\n📊 Configuration:")
    print(f"   Episodes: {episodes}")
    print(f"   Timesteps: {timesteps:,}")
    print(f"   Algorithm: PPO (Proximal Policy Optimization)")
    print(f"   Offensive Actions: 10")
    print()
    
    # Train the Red Agent
    print("🔄 Training Red Agent...\n")
    start_time = time.time()
    
    model.learn(total_timesteps=timesteps, progress_bar=True)
    
    end_time = time.time()
    duration = end_time - start_time
    
    # Save model
    model_path = "models/aceac_red_agent_100ep.zip"
    model.save(model_path)
    
    print("\n🧪 Testing trained Red Agent...")
    
    # Test the trained agent
    episode_rewards = []
    success_rates = []
    final_securities = []
    
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
        success_rates.append(info['success_rate'])
        final_securities.append(info['network_security'])
        
        print(f"  Test Episode {ep+1}: "
              f"Reward={ep_reward:.1f}, "
              f"Success={info['success_rate']:.1%}, "
              f"Final Security={info['network_security']:.2f}")
    
    avg_reward = sum(episode_rewards) / len(episode_rewards)
    avg_success = sum(success_rates) / len(success_rates)
    avg_security = sum(final_securities) / len(final_securities)
    
    # Save log
    log_data = {
        "agent_type": "red_offensive",
        "episodes": episodes,
        "total_timesteps": timesteps,
        "algorithm": "PPO",
        "training_location": "kali_linux_virtualbox",
        "user": "sarowarzahan414",
        "date": "2025-10-08",
        "time_utc": "22:05:30",
        "start_time": datetime.fromtimestamp(start_time, timezone.utc).isoformat(),
        "end_time": datetime.fromtimestamp(end_time, timezone.utc).isoformat(),
        "duration_seconds": duration,
        "timesteps_per_second": timesteps / duration,
        "model_path": model_path,
        "test_episodes": 10,
        "test_rewards": episode_rewards,
        "avg_test_reward": avg_reward,
        "avg_success_rate": avg_success,
        "avg_final_security": avg_security,
        "status": "validation_complete",
        "bug_fixed": "numpy_array_to_int_conversion"
    }
    
    log_path = "logs/aceac_red_100ep_validation.json"
    with open(log_path, "w") as f:
        json.dump(log_data, f, indent=2)
    
    # Results
    print("\n" + "="*70)
    print("✅ VALIDATION TRAINING COMPLETE!")
    print("="*70)
    print(f"Duration: {duration:.2f} seconds")
    print(f"Speed: {timesteps/duration:.0f} timesteps/second")
    print(f"Avg Test Reward: {avg_reward:.2f}")
    print(f"Avg Success Rate: {avg_success:.1%}")
    print(f"Avg Final Security: {avg_security:.2f} (lower = more damage)")
    print(f"Model: {model_path}")
    print(f"Log: {log_path}")
    print("="*70)
    print("\n🎯 Ready for Red vs Blue co-evolution!")
    print("="*70 + "\n")
    
    return model, log_data


if __name__ == "__main__":
    print("🚀 Starting ACEAC Red Agent Training (FIXED VERSION)...")
    model, log = train_red_agent_100ep()
    print("🎉 Red Agent validation complete!")
    print("📋 Next: Run co-evolution training (aceac_coevolution_1000ep.py)")
