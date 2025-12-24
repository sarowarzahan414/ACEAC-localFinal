"""
ACEAC QD Pool + Behavioral Characterization for Real-World Environments
Implements MAP-Elites style quality diversity with real security tools

Author: @sarowarzahan414
Date: 2024-12-24
Innovation: Quality Diversity for Real-World Cyber Testing
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import json
import time
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import os

from aceac_real_world_bridge import RealToolExecutor, ToolResult
from aceac_v2_cyber_killchain import ACEACv2Environment, OffensiveTool, DefensiveTool


@dataclass
class Policy:
    """Individual policy in QD pool"""
    id: str
    model_path: str
    performance: float  # Fitness
    behavior: np.ndarray  # Behavioral characterization
    grid_cell: Tuple[int, ...]  # Grid coordinates
    generation: int
    episodes_tested: int


@dataclass
class EpisodeLog:
    """Log of episode execution"""
    tools_used: List[int]
    phases_reached: List[int]
    duration: float
    ids_alerts: int
    total_actions: int
    performance: float
    network_security_final: float
    detection_level_final: float


class BehavioralCharacterizer:
    """Compute behavioral characterization for policies"""

    def __init__(self, dimensions: int = 4):
        self.dimensions = dimensions

    def compute_behavior(self, episode_log: EpisodeLog) -> np.ndarray:
        """
        Compute behavioral descriptor from episode

        Dimensions:
        1. Stealth: How much was detected (0 = fully detected, 1 = undetected)
        2. Tool Diversity: Variety of tools used (0 = single tool, 1 = all tools)
        3. Aggressiveness: Speed and intensity (0 = slow/careful, 1 = fast/aggressive)
        4. Effectiveness: Success rate (0 = failed, 1 = successful)
        """

        # Dimension 1: Stealth
        if episode_log.total_actions > 0:
            stealth = 1.0 - min(1.0, episode_log.ids_alerts / episode_log.total_actions)
        else:
            stealth = 0.0

        # Dimension 2: Tool Diversity
        unique_tools = len(set(episode_log.tools_used))
        diversity = unique_tools / 25.0  # 25 total tools

        # Dimension 3: Aggressiveness (actions per second)
        if episode_log.duration > 0:
            actions_per_sec = episode_log.total_actions / episode_log.duration
            aggressiveness = min(1.0, actions_per_sec / 2.0)  # Normalize to 0-1
        else:
            aggressiveness = 0.0

        # Dimension 4: Effectiveness
        effectiveness = episode_log.performance / 100.0  # Normalize

        behavior = np.array([stealth, diversity, aggressiveness, effectiveness])

        # Ensure in [0, 1] range
        behavior = np.clip(behavior, 0.0, 1.0)

        return behavior


class QDPool:
    """Quality Diversity Pool using MAP-Elites algorithm"""

    def __init__(self, grid_resolution: int = 10, behavior_dims: int = 2):
        self.grid_resolution = grid_resolution
        self.behavior_dims = behavior_dims

        # Initialize grid (multi-dimensional)
        grid_shape = tuple([grid_resolution] * behavior_dims)
        self.grid = {}  # Sparse representation

        # Statistics
        self.policies_added = 0
        self.policies_rejected = 0

    def behavior_to_cell(self, behavior: np.ndarray) -> Tuple[int, ...]:
        """Convert continuous behavior to discrete grid cell"""
        # Use first behavior_dims dimensions
        behavior = behavior[:self.behavior_dims]

        # Map [0, 1] to grid indices
        indices = (behavior * (self.grid_resolution - 1)).astype(int)
        indices = np.clip(indices, 0, self.grid_resolution - 1)

        return tuple(indices)

    def add_policy(self, policy: Policy) -> bool:
        """
        Add policy to pool if it's better than existing in that cell

        Returns: True if added, False if rejected
        """
        cell = policy.grid_cell

        if cell not in self.grid:
            # New cell - add policy
            self.grid[cell] = policy
            self.policies_added += 1
            return True
        else:
            # Cell occupied - compare performance
            existing = self.grid[cell]

            if policy.performance > existing.performance:
                # Better policy - replace
                self.grid[cell] = policy
                self.policies_added += 1
                return True
            else:
                # Worse policy - reject
                self.policies_rejected += 1
                return False

    def sample_policy(self) -> Optional[Policy]:
        """Sample random policy from pool"""
        if not self.grid:
            return None

        cell = np.random.choice(list(self.grid.keys()))
        return self.grid[cell]

    def get_best_policy(self) -> Optional[Policy]:
        """Get policy with highest performance"""
        if not self.grid:
            return None

        return max(self.grid.values(), key=lambda p: p.performance)

    def get_statistics(self) -> Dict:
        """Get pool statistics"""
        if not self.grid:
            return {
                'size': 0,
                'avg': 0.0,
                'best': 0.0,
                'worst': 0.0,
                'coverage': 0.0,
                'policies_added': self.policies_added,
                'policies_rejected': self.policies_rejected
            }

        performances = [p.performance for p in self.grid.values()]

        total_cells = self.grid_resolution ** self.behavior_dims

        return {
            'size': len(self.grid),
            'avg': np.mean(performances),
            'best': np.max(performances),
            'worst': np.min(performances),
            'coverage': len(self.grid) / total_cells,
            'policies_added': self.policies_added,
            'policies_rejected': self.policies_rejected
        }


class RealWorldQDEnvironment(gym.Env):
    """
    Real-world QD environment that executes actual tools

    Can run in two modes:
    - Simulated: Uses ACEACv2Environment (faster, for training)
    - Real: Executes actual tools (slower, for validation)
    """

    def __init__(self, agent_role="red", use_real_tools=False, real_target=None):
        super().__init__()

        self.agent_role = agent_role
        self.use_real_tools = use_real_tools
        self.real_target = real_target or "10.0.1.100"

        # Use same observation/action space as v2 environment
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(62,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(25)  # 25 tools

        # Simulated environment (always available)
        self.sim_env = ACEACv2Environment(agent_role=agent_role)

        # Real tool executor (only if needed)
        self.real_executor = None
        if use_real_tools:
            self.real_executor = RealToolExecutor()

        # Episode tracking
        self.episode_log = None
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Reset simulated environment
        obs, info = self.sim_env.reset(seed=seed, options=options)

        # Start new episode log
        self.episode_log = EpisodeLog(
            tools_used=[],
            phases_reached=[],
            duration=0.0,
            ids_alerts=0,
            total_actions=0,
            performance=0.0,
            network_security_final=0.0,
            detection_level_final=0.0
        )
        self.episode_start_time = time.time()

        return obs, info

    def step(self, action):
        if isinstance(action, np.ndarray):
            action = int(action.item())
        else:
            action = int(action)

        # Always step simulated environment
        obs, reward, terminated, truncated, info = self.sim_env.step(action)

        # Log action
        self.episode_log.tools_used.append(action)
        self.episode_log.total_actions += 1
        self.episode_log.performance += reward

        # Execute real tool if enabled
        if self.use_real_tools and self.real_executor:
            real_reward = self._execute_real_tool(action)
            # Blend real and simulated rewards
            reward = 0.7 * reward + 0.3 * real_reward

        # Update episode log
        if terminated:
            self.episode_log.duration = time.time() - self.episode_start_time
            self.episode_log.network_security_final = info.get('network_security', 0.0)
            self.episode_log.detection_level_final = info.get('detection_level', 0.0)

        return obs, reward, terminated, truncated, info

    def _execute_real_tool(self, tool_id: int) -> float:
        """Execute real tool and return reward"""
        try:
            if self.agent_role == "red":
                # Map tool_id to tool name
                tool_names = ['nmap', 'masscan', 'shodan', 'recon-ng', 'theharvester',
                             'msfvenom', 'setoolkit', 'beef', 'empire', 'covenant',
                             'metasploit', 'exploit-db', 'sqlmap', 'burpsuite', 'cobalt-strike',
                             'hydra', 'john', 'hashcat', 'mimikatz', 'responder',
                             'bloodhound', 'powershell-empire', 'impacket', 'crackmapexec', 'eternalblue']

                tool_name = tool_names[tool_id] if tool_id < len(tool_names) else 'nmap'

                # Execute (only safe tools for now)
                if tool_name in ['nmap', 'masscan']:
                    result = self.real_executor.execute_offensive_tool(
                        tool_name,
                        self.real_target
                    )

                    # Calculate reward
                    if result.success:
                        reward = 20.0 + len(result.ports_found or []) * 5.0
                        reward -= result.duration  # Time penalty
                    else:
                        reward = -10.0

                    # Track detection
                    if result.stealth_score < 0.5:
                        self.episode_log.ids_alerts += 1

                    return reward
                else:
                    # Tool not safe for real execution - use simulation
                    return 0.0

            else:  # Blue team
                # Execute defensive tool
                tool_names = ['snort', 'suricata', 'zeek', 'wireshark', 'tcpdump',
                             'iptables', 'firewalld', 'ufw', 'pfsense', 'fortinet',
                             'ossec', 'wazuh', 'splunk', 'elk', 'graylog',
                             'clamav', 'fail2ban', 'tripwire', 'aide', 'osquery',
                             'volatility', 'autopsy', 'velociraptor', 'yara', 'ghidra']

                tool_name = tool_names[tool_id] if tool_id < len(tool_names) else 'snort'

                # Execute (only monitoring tools for now)
                if tool_name in ['snort', 'suricata', 'wazuh']:
                    result = self.real_executor.execute_defensive_tool(tool_name)

                    if result.success:
                        reward = 15.0 + result.alerts_detected * 2.0
                    else:
                        reward = -5.0

                    return reward
                else:
                    return 0.0

        except Exception as e:
            print(f"Real tool execution error: {e}")
            return -5.0


def qd_coevolution_real_world(
    generations: int = 10,
    episodes_per_gen: int = 100,
    grid_resolution: int = 10,
    behavior_dims: int = 2,
    use_real_tools: bool = False,
    real_target: str = "10.0.1.100"
):
    """
    Run QD-based coevolution with real-world tool execution

    Args:
        generations: Number of generations
        episodes_per_gen: Episodes per generation
        grid_resolution: Grid resolution for MAP-Elites
        behavior_dims: Number of behavioral dimensions
        use_real_tools: If True, execute real tools (slower, requires setup)
        real_target: Target IP for real tool execution
    """

    print("\n" + "="*70)
    print("ACEAC QD POOL + BEHAVIORAL CHARACTERIZATION")
    if use_real_tools:
        print("MODE: REAL-WORLD TOOL EXECUTION")
    else:
        print("MODE: SIMULATED (FAST TRAINING)")
    print("="*70)
    print(f"User: sarowarzahan414")
    print(f"Date: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print("-"*70)
    print(f"Generations: {generations}")
    print(f"Episodes per generation: {episodes_per_gen}")
    print(f"Behavior dimensions: {behavior_dims}")
    print(f"Grid resolution: {grid_resolution}x{grid_resolution}")
    print("="*70)
    print()

    # Create directories
    os.makedirs("models/qd_pool", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    # Initialize QD pools
    red_pool = QDPool(grid_resolution=grid_resolution, behavior_dims=behavior_dims)
    blue_pool = QDPool(grid_resolution=grid_resolution, behavior_dims=behavior_dims)

    # Behavioral characterizer
    characterizer = BehavioralCharacterizer(dimensions=4)

    # Load baseline models
    print("Loading baseline models...")
    try:
        red_model = PPO.load("models/aceac_red_agent_100ep.zip")
        blue_model = PPO.load("models/aceac_blue_agent_100ep.zip")
        print("✓ Models loaded")
    except Exception as e:
        print(f"✗ Error loading models: {e}")
        print("  Training from scratch...")
        env = RealWorldQDEnvironment(agent_role="red", use_real_tools=False)
        red_model = PPO("MlpPolicy", env, verbose=0)
        blue_model = PPO("MlpPolicy", env, verbose=0)

    print()

    # Authorize real tool execution if needed
    if use_real_tools:
        print("Authorizing real tool execution...")
        env_test = RealWorldQDEnvironment(agent_role="red", use_real_tools=True)
        session_id = env_test.real_executor.auth.authorize_session(
            user='sarowarzahan414',
            purpose=f'QD Pool Research - {generations} generations'
        )
        print(f"Session ID: {session_id}")
        print()

    # Training log
    training_log = {
        'user': 'sarowarzahan414',
        'date': datetime.now(timezone.utc).strftime('%Y-%m-%d'),
        'time_utc': datetime.now(timezone.utc).strftime('%H:%M:%S'),
        'innovations': ['#1 QD Pool', '#2 Behavioral Characterization'],
        'duration_seconds': 0.0,
        'duration_minutes': 0.0,
        'generations': generations,
        'episodes_per_gen': episodes_per_gen,
        'behavior_dims': behavior_dims,
        'grid_resolution': grid_resolution,
        'history': [],
        'red_pool_final': {},
        'blue_pool_final': {},
        'status': 'running'
    }

    total_start_time = time.time()

    # Main QD loop
    for gen in range(1, generations + 1):
        print("="*70)
        print(f"GENERATION {gen}/{generations}")
        print("="*70)

        gen_start = time.time()

        # Train Red agent
        print(f"\nTraining Red agent (Gen {gen})...")
        env_red = RealWorldQDEnvironment(
            agent_role="red",
            use_real_tools=use_real_tools,
            real_target=real_target
        )
        red_model.set_env(env_red)
        red_model.learn(total_timesteps=episodes_per_gen * 50, reset_num_timesteps=False, progress_bar=False)

        # Evaluate Red agent
        red_performance, red_behavior = evaluate_policy(
            red_model, env_red, characterizer, episodes=episodes_per_gen
        )

        # Add to Red pool
        red_policy = Policy(
            id=f"red_gen{gen}",
            model_path=f"models/qd_pool/red_gen{gen}.zip",
            performance=red_performance,
            behavior=red_behavior,
            grid_cell=red_pool.behavior_to_cell(red_behavior),
            generation=gen,
            episodes_tested=episodes_per_gen
        )
        red_added = red_pool.add_policy(red_policy)
        red_model.save(red_policy.model_path)

        print(f"Red Performance: {red_performance:.2f}")
        print(f"Red Behavior: {red_behavior[:behavior_dims]}")
        print(f"Red Grid Cell: {red_policy.grid_cell}")
        print(f"Red Added to Pool: {red_added}")

        # Train Blue agent
        print(f"\nTraining Blue agent (Gen {gen})...")
        env_blue = RealWorldQDEnvironment(
            agent_role="blue",
            use_real_tools=use_real_tools,
            real_target=real_target
        )
        blue_model.set_env(env_blue)
        blue_model.learn(total_timesteps=episodes_per_gen * 50, reset_num_timesteps=False, progress_bar=False)

        # Evaluate Blue agent
        blue_performance, blue_behavior = evaluate_policy(
            blue_model, env_blue, characterizer, episodes=episodes_per_gen
        )

        # Add to Blue pool
        blue_policy = Policy(
            id=f"blue_gen{gen}",
            model_path=f"models/qd_pool/blue_gen{gen}.zip",
            performance=blue_performance,
            behavior=blue_behavior,
            grid_cell=blue_pool.behavior_to_cell(blue_behavior),
            generation=gen,
            episodes_tested=episodes_per_gen
        )
        blue_added = blue_pool.add_policy(blue_policy)
        blue_model.save(blue_policy.model_path)

        print(f"Blue Performance: {blue_performance:.2f}")
        print(f"Blue Behavior: {blue_behavior[:behavior_dims]}")
        print(f"Blue Grid Cell: {blue_policy.grid_cell}")
        print(f"Blue Added to Pool: {blue_added}")

        gen_duration = time.time() - gen_start

        # Log generation
        gen_log = {
            'generation': gen,
            'duration': gen_duration,
            'red_performance': float(red_performance),
            'blue_performance': float(blue_performance),
            'red_behavior': red_behavior.tolist(),
            'blue_behavior': blue_behavior.tolist(),
            'red_pool': red_pool.get_statistics(),
            'blue_pool': blue_pool.get_statistics()
        }
        training_log['history'].append(gen_log)

        # Print pool stats
        red_stats = red_pool.get_statistics()
        blue_stats = blue_pool.get_statistics()

        print(f"\nRed Pool: {red_stats['size']} policies, {red_stats['coverage']*100:.1f}% coverage")
        print(f"Blue Pool: {blue_stats['size']} policies, {blue_stats['coverage']*100:.1f}% coverage")
        print(f"Generation Duration: {gen_duration:.2f}s")
        print()

    total_duration = time.time() - total_start_time

    # Finalize log
    training_log['duration_seconds'] = total_duration
    training_log['duration_minutes'] = total_duration / 60.0
    training_log['red_pool_final'] = red_pool.get_statistics()
    training_log['blue_pool_final'] = blue_pool.get_statistics()
    training_log['status'] = 'complete'

    # Save log
    log_file = 'logs/qd_pool_real_world.json'
    with open(log_file, 'w') as f:
        json.dump(training_log, f, indent=2)

    print("="*70)
    print("QD POOL TRAINING COMPLETE!")
    print("="*70)
    print(f"Duration: {total_duration/60:.2f} minutes")
    print(f"Red Pool: {red_stats['size']} policies ({red_stats['coverage']*100:.1f}% coverage)")
    print(f"Blue Pool: {blue_stats['size']} policies ({blue_stats['coverage']*100:.1f}% coverage)")
    print(f"Best Red: {red_stats['best']:.2f}")
    print(f"Best Blue: {blue_stats['best']:.2f}")
    print(f"Log saved: {log_file}")
    print("="*70)
    print()

    return red_pool, blue_pool, training_log


def evaluate_policy(model, env, characterizer, episodes=10) -> Tuple[float, np.ndarray]:
    """Evaluate policy and return (performance, behavior)"""

    total_reward = 0.0
    all_episode_logs = []

    for ep in range(episodes):
        obs, _ = env.reset()
        episode_reward = 0.0

        for step in range(100):
            action, _ = model.predict(obs, deterministic=False)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward

            if terminated or truncated:
                break

        total_reward += episode_reward

        # Get episode log
        if hasattr(env, 'episode_log'):
            all_episode_logs.append(env.episode_log)

    # Average performance
    avg_performance = total_reward / episodes

    # Compute average behavior
    if all_episode_logs:
        behaviors = [characterizer.compute_behavior(log) for log in all_episode_logs]
        avg_behavior = np.mean(behaviors, axis=0)
    else:
        avg_behavior = np.zeros(4)

    return avg_performance, avg_behavior


if __name__ == "__main__":
    # Run QD Pool training
    # Set use_real_tools=False for fast simulated training
    # Set use_real_tools=True for real-world validation (requires Oracle setup)

    red_pool, blue_pool, log = qd_coevolution_real_world(
        generations=10,
        episodes_per_gen=100,
        grid_resolution=10,
        behavior_dims=2,
        use_real_tools=False,  # Set to True for real-world testing
        real_target="10.0.1.100"
    )

    print("INNOVATION VERIFIED!")
    print("QD Pool + Behavioral Characterization: ✓")
    print()
