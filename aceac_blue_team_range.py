"""
ACEAC Cyber Range - Red Team Environment
Offensive agent training environment

Author: @sarowarzahan414
Date: 2025-10-08 22:39:12 UTC
Location: Kali Linux VirtualBox
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np


class ACEACCyberRange(gym.Env):
    """Red Team Offensive Training Environment"""
    
    metadata = {'render_modes': [], 'name': 'ACEAC-CyberRange-Red-v1'}
    
    def __init__(self):
        super().__init__()
        
        # Observation: Network state (20 features)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(20,), dtype=np.float32
        )
        
        # Action: 10 offensive actions
        # 0: Scan, 1: Probe, 2: Exploit, 3: Privilege Escalation
        # 4: Lateral Movement, 5: Exfiltration, 6: Persistence
        # 7: C2, 8: Ransomware, 9: Cover Tracks
        self.action_space = spaces.Discrete(10)
        
        self.max_steps = 50
        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.network_security = 0.8
        self.successful_attacks = 0
        self.total_attempts = 0
        return self._get_observation(), {}
    
    def step(self, action):
        self.current_step += 1
        self.total_attempts += 1
        
        if isinstance(action, np.ndarray):
            action = int(action.item())
        else:
            action = int(action)
        
        # Attack mechanics
        attack_power = self._get_attack_power(action)
        success_prob = attack_power * (1.2 - self.network_security)
        
        if np.random.random() < success_prob:
            reward = 15.0 + attack_power * 20.0
            self.network_security = max(0.0, self.network_security - attack_power)
            self.successful_attacks += 1
        else:
            reward = -8.0
        
        # Stealth penalties
        if action in [2, 3, 8]:
            reward -= 2.0
        
        terminated = self.current_step >= self.max_steps
        
        info = {
            'network_security': float(self.network_security),
            'successful_attacks': int(self.successful_attacks),
            'success_rate': float(self.successful_attacks / self.total_attempts)
        }
        
        return self._get_observation(), float(reward), terminated, False, info
    
    def _get_attack_power(self, action):
        powers = {
            0: 0.08, 1: 0.12, 2: 0.18, 3: 0.22, 4: 0.15,
            5: 0.16, 6: 0.14, 7: 0.10, 8: 0.20, 9: 0.09
        }
        return float(powers[action] + np.random.uniform(-0.02, 0.02))
    
    def _get_observation(self):
        obs = np.zeros(20, dtype=np.float32)
        obs[0] = self.network_security
        obs[1] = self.successful_attacks / (self.total_attempts + 1)
        obs[2] = self.current_step / self.max_steps
        obs[3:] = np.random.random(17) * 0.5
        return obs


if __name__ == "__main__":
    env = ACEACCyberRange()
    print("ACEAC Red Team Cyber Range Environment")
    print("Observation space:", env.observation_space.shape)
    print("Action space:", env.action_space.n)
    
    obs, _ = env.reset()
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {i+1}: Action={action}, Reward={reward:.1f}, Security={info['network_security']:.2f}")
        if terminated:
            break
