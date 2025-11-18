"""
ACEAC v2.0 - Validation Script (CLEAN)
Test Cyber Kill Chain and tool effectiveness

Author: @sarowarzahan414
Date: 2025-10-08 23:47:52 UTC
Location: Kali Linux VirtualBox
"""

import numpy as np
from stable_baselines3 import PPO
from aceac_v2_cyber_killchain import ACEACv2Environment
import json
from collections import defaultdict


def validate_agent(model_path, agent_role="red", num_episodes=20):
    """Validate agent performance"""
    
    print("")
    print("="*70)
    print("Validating " + agent_role.upper() + " Agent")
    print("="*70)
    print("Model: " + model_path)
    print("Episodes: " + str(num_episodes))
    print("="*70)
    
    try:
        model = PPO.load(model_path)
        env = ACEACv2Environment(agent_role=agent_role)
    except Exception as e:
        print("Error loading model: " + str(e))
        return None
    
    tool_usage = defaultdict(int)
    episode_rewards = []
    kill_chain_progress = []
    final_phases = []
    network_securities = []
    
    for ep in range(num_episodes):
        obs, _ = env.reset()
        ep_reward = 0.0
        
        for step in range(100):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            
            ep_reward += reward
            tool_usage[int(action)] += 1
            
            if terminated:
                break
        
        episode_rewards.append(ep_reward)
        kill_chain_progress.append(info['kill_chain_progress'])
        final_phases.append(info['current_phase'])
        network_securities.append(info['network_security'])
        
        print("Episode " + str(ep+1) + ": Reward=" + str(round(ep_reward, 1)) + 
              ", Phase=" + info['current_phase'] + 
              ", Progress=" + str(round(info['kill_chain_progress']*100, 1)) + "%")
    
    # Calculate statistics
    avg_reward = sum(episode_rewards) / len(episode_rewards)
    avg_progress = sum(kill_chain_progress) / len(kill_chain_progress)
    avg_security = sum(network_securities) / len(network_securities)
    
    # Phase distribution
    phase_counts = defaultdict(int)
    for phase in final_phases:
        phase_counts[phase] += 1
    
    print("")
    print("="*70)
    print("RESULTS")
    print("="*70)
    print("Average Reward: " + str(round(avg_reward, 2)))
    print("Avg Kill Chain Progress: " + str(round(avg_progress*100, 1)) + "%")
    print("Avg Network Security: " + str(round(avg_security, 2)))
    
    print("")
    print("Final Phase Distribution:")
    for phase, count in sorted(phase_counts.items()):
        pct = (count / num_episodes) * 100
        print("  " + phase + ": " + str(count) + "/" + str(num_episodes) + 
              " (" + str(round(pct, 1)) + "%)")
    
    print("")
    print("Top 10 Tools Used:")
    sorted_tools = sorted(tool_usage.items(), key=lambda x: x[1], reverse=True)
    for tool_id, count in sorted_tools[:10]:
        print("  Tool " + str(tool_id) + ": " + str(count) + " times")
    
    print("="*70)
    
    results = {
        'agent_role': agent_role,
        'model_path': model_path,
        'num_episodes': num_episodes,
        'avg_reward': float(avg_reward),
        'avg_kill_chain_progress': float(avg_progress),
        'avg_network_security': float(avg_security),
        'episode_rewards': [float(r) for r in episode_rewards],
        'kill_chain_progress': [float(p) for p in kill_chain_progress],
        'phase_distribution': {k: int(v) for k, v in phase_counts.items()},
        'tool_usage': {str(k): int(v) for k, v in tool_usage.items()}
    }
    
    return results


def compare_models():
    """Compare v1 vs v2 models"""
    
    print("")
    print("="*70)
    print("ACEAC v1 vs v2 COMPARISON")
    print("="*70)
    
    print("")
    print("v1.0 Features:")
    print("  - Observation: 20D")
    print("  - Actions: 10 per agent")
    print("  - Training: Basic co-evolution")
    
    print("")
    print("v2.0 Features:")
    print("  - Observation: 62D")
    print("  - Actions: 25 real tools per agent")
    print("  - Training: Cyber Kill Chain + SWAP RL")
    
    print("")
    print("="*70)


def main():
    """Main validation"""
    
    print("")
    print("="*70)
    print("ACEAC v2.0 VALIDATION")
    print("="*70)
    print("User: sarowarzahan414")
    print("Date: 2025-10-08 23:47:52 UTC")
    print("="*70)
    
    # Validate Red Agent
    red_results = validate_agent(
        model_path="models/aceac_v2/red_final.zip",
        agent_role="red",
        num_episodes=20
    )
    
    # Validate Blue Agent
    blue_results = validate_agent(
        model_path="models/aceac_v2/blue_final.zip",
        agent_role="blue",
        num_episodes=20
    )
    
    # Compare models
    compare_models()
    
    # Save results
    if red_results and blue_results:
        validation_log = {
            'user': 'sarowarzahan414',
            'date': '2025-10-08',
            'time_utc': '23:47:52',
            'red_agent': red_results,
            'blue_agent': blue_results
        }
        
        with open('logs/aceac_v2_validation.json', 'w') as f:
            json.dump(validation_log, f, indent=2)
        
        print("")
        print("Validation log saved: logs/aceac_v2_validation.json")
    
    print("")
    print("="*70)
    print("VALIDATION COMPLETE!")
    print("="*70)
    print("")


if __name__ == "__main__":
    main()
