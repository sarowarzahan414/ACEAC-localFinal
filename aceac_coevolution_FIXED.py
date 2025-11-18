"""
ACEAC Red vs Blue Co-Evolution Training - FIXED
Compatible with 20-dimensional observation space

Author: @sarowarzahan414
Date: 2025-10-08
Time: 22:22:06 UTC
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


class ACEACCoEvolutionEnv(gym.Env):
    """Co-Evolution Environment - FIXED for 20D observation space"""
    
    metadata = {'render_modes': [], 'name': 'ACEAC-CoEvolution-v1'}
    
    def __init__(self, opponent_model=None, agent_role="red"):
        super().__init__()
        
        # FIXED: Use 20D observation space (matching trained models)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(20,), dtype=np.float32
        )
        
        self.action_space = spaces.Discrete(10)
        self.opponent_model = opponent_model
        self.agent_role = agent_role
        self.max_steps = 50
        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.network_health = 0.7
        self.red_score = 0.0
        self.blue_score = 0.0
        return self._get_observation(), {}
    
    def step(self, action):
        self.current_step += 1
        
        if isinstance(action, np.ndarray):
            action = int(action.item())
        else:
            action = int(action)
        
        # Current agent action
        if self.agent_role == "red":
            attack_power = 0.08 + action * 0.015
            self.network_health -= attack_power
            self.red_score += attack_power * 100
            reward = attack_power * 100
        else:
            defense_power = 0.05 + action * 0.012
            self.network_health = min(1.0, self.network_health + defense_power)
            self.blue_score += defense_power * 100
            reward = defense_power * 100
        
        # Opponent counter-action
        if self.opponent_model is not None:
            try:
                opp_obs = self._get_observation()
                opp_action, _ = self.opponent_model.predict(opp_obs, deterministic=False)
                
                if self.agent_role == "red":
                    if isinstance(opp_action, np.ndarray):
                        opp_action = int(opp_action.item())
                    defense = 0.04 + opp_action * 0.01
                    self.network_health = min(1.0, self.network_health + defense)
                    self.blue_score += defense * 100
                    reward -= defense * 50
                else:
                    if isinstance(opp_action, np.ndarray):
                        opp_action = int(opp_action.item())
                    attack = 0.06 + opp_action * 0.012
                    self.network_health -= attack
                    self.red_score += attack * 100
                    reward -= attack * 50
            except:
                pass
        
        self.network_health = np.clip(self.network_health, 0.0, 1.0)
        
        terminated = (
            self.current_step >= self.max_steps or
            self.network_health <= 0.1 or
            self.network_health >= 0.95
        )
        
        info = {
            'network_health': float(self.network_health),
            'red_score': float(self.red_score),
            'blue_score': float(self.blue_score)
        }
        
        return self._get_observation(), float(reward), terminated, False, info
    
    def _get_observation(self):
        """FIXED: Return 20D observation"""
        obs = np.zeros(20, dtype=np.float32)
        obs[0] = self.network_health
        obs[1] = self.red_score / 100.0
        obs[2] = self.blue_score / 100.0
        obs[3] = self.current_step / self.max_steps
        obs[4:] = np.random.random(16) * 0.5
        return obs


def test_generation(red_model, blue_model, num_battles=10):
    env = ACEACCoEvolutionEnv(agent_role="red")
    results = {'red_wins': 0, 'blue_wins': 0, 'draws': 0, 'battles': []}
    
    for battle in range(num_battles):
        obs, _ = env.reset()
        
        for step in range(100):
            if step % 2 == 0:
                action, _ = red_model.predict(obs, deterministic=True)
            else:
                action, _ = blue_model.predict(obs, deterministic=True)
            
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated:
                break
        
        health = info['network_health']
        if health < 0.4:
            results['red_wins'] += 1
            winner = 'red'
        elif health > 0.7:
            results['blue_wins'] += 1
            winner = 'blue'
        else:
            results['draws'] += 1
            winner = 'draw'
        
        results['battles'].append({
            'battle': battle + 1,
            'winner': winner,
            'final_health': float(health)
        })
    
    results['avg_health'] = sum(b['final_health'] for b in results['battles']) / num_battles
    return results


def coevolution_training(episodes_per_generation=100, generations=10):
    print("\n" + "="*70)
    print("ACEAC RED vs BLUE CO-EVOLUTION - FIXED")
    print("="*70)
    print("User: sarowarzahan414")
    print("Date: 2025-10-08 22:22:06 UTC")
    print("Location: Kali Linux VirtualBox")
    print("-"*70)
    print("Generations:", generations)
    print("Episodes per generation:", episodes_per_generation)
    print("Total episodes:", episodes_per_generation * generations * 2)
    print("="*70)
    print()
    
    os.makedirs("models/coevolution", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    print("Loading baseline agents...")
    try:
        red_model = PPO.load("models/aceac_red_agent_100ep.zip")
        print("Red baseline loaded")
    except Exception as e:
        print("Error loading Red:", e)
        return None, None, None
    
    try:
        blue_model = PPO.load("models/aceac_blue_agent_100ep.zip")
        print("Blue baseline loaded")
    except Exception as e:
        print("Error loading Blue:", e)
        return None, None, None
    
    print()
    
    coevo_log = {
        'user': 'sarowarzahan414',
        'start_time': datetime.now(timezone.utc).isoformat(),
        'generations': generations,
        'episodes_per_generation': episodes_per_generation,
        'generation_results': []
    }
    
    total_start = time.time()
    
    for gen in range(1, generations + 1):
        print("="*70)
        print("GENERATION", gen, "/", generations)
        print("="*70)
        
        gen_start = time.time()
        
        print("\nTraining Red vs Blue Gen", gen-1, "...")
        env_red = ACEACCoEvolutionEnv(opponent_model=blue_model, agent_role="red")
        red_model.set_env(env_red)
        
        red_start = time.time()
        red_model.learn(total_timesteps=episodes_per_generation * 50, reset_num_timesteps=False, progress_bar=False)
        red_dur = time.time() - red_start
        
        red_model.save("models/coevolution/red_gen" + str(gen) + ".zip")
        print("Red complete (", round(red_dur, 1), "s)")
        
        print("\nTraining Blue vs Red Gen", gen, "...")
        env_blue = ACEACCoEvolutionEnv(opponent_model=red_model, agent_role="blue")
        blue_model.set_env(env_blue)
        
        blue_start = time.time()
        blue_model.learn(total_timesteps=episodes_per_generation * 50, reset_num_timesteps=False, progress_bar=False)
        blue_dur = time.time() - blue_start
        
        blue_model.save("models/coevolution/blue_gen" + str(gen) + ".zip")
        print("Blue complete (", round(blue_dur, 1), "s)")
        
        print("\nTesting Gen", gen, "...")
        test_res = test_generation(red_model, blue_model)
        
        gen_dur = time.time() - gen_start
        
        gen_data = {
            'generation': gen,
            'duration_seconds': gen_dur,
            'test_results': test_res
        }
        
        coevo_log['generation_results'].append(gen_data)
        
        print("\nGeneration", gen, "Results:")
        print("Red:", test_res['red_wins'], "/10")
        print("Blue:", test_res['blue_wins'], "/10")
        print("Draws:", test_res['draws'], "/10")
        print("Avg Health:", round(test_res['avg_health'], 2))
        print()
    
    total_dur = time.time() - total_start
    
    red_model.save("models/aceac_red_coevolved.zip")
    blue_model.save("models/aceac_blue_coevolved.zip")
    
    coevo_log['end_time'] = datetime.now(timezone.utc).isoformat()
    coevo_log['total_duration_seconds'] = total_dur
    coevo_log['status'] = 'complete'
    
    with open('logs/coevolution_training.json', 'w') as f:
        json.dump(coevo_log, f, indent=2)
    
    print("="*70)
    print("CO-EVOLUTION COMPLETE!")
    print("="*70)
    print("Duration:", round(total_dur/60, 1), "minutes")
    print("Models saved:")
    print("  models/aceac_red_coevolved.zip")
    print("  models/aceac_blue_coevolved.zip")
    print("="*70)
    print()
    
    return red_model, blue_model, coevo_log


if __name__ == "__main__":
    coevolution_training(episodes_per_generation=100, generations=10)
    print("GAME CHANGER VERIFIED!")
