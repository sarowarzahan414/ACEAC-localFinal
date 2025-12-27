"""
ACEAC v2.0 - Advanced Cyber Kill Chain + Real Tools
Implements Cyber Kill Chain, SWAP RL, and Real-World Tools

Author: @sarowarzahan414
Date: 2025-10-08 23:14:00 UTC
Location: Kali Linux VirtualBox
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from enum import Enum
from typing import Dict, List, Tuple
import json


class CyberKillChainPhase(Enum):
    """Cyber Kill Chain Phases"""
    RECONNAISSANCE = 0
    WEAPONIZATION = 1
    DELIVERY = 2
    EXPLOITATION = 3
    INSTALLATION = 4
    COMMAND_CONTROL = 5
    ACTIONS_OBJECTIVES = 6


class OffensiveTool(Enum):
    """Real-World Offensive Tools (25 tools)"""
    # Reconnaissance (0-4)
    NMAP = 0
    MASSCAN = 1
    SHODAN = 2
    RECON_NG = 3
    THEHARVESTER = 4
    
    # Weaponization (5-9)
    MSFVENOM = 5
    SETOOLKIT = 6
    BEEF = 7
    EMPIRE = 8
    COVENANT = 9
    
    # Exploitation (10-14)
    METASPLOIT = 10
    EXPLOIT_DB = 11
    SQLMAP = 12
    BURPSUITE = 13
    COBALT_STRIKE = 14
    
    # Password Attacks (15-19)
    HYDRA = 15
    JOHN = 16
    HASHCAT = 17
    MIMIKATZ = 18
    RESPONDER = 19
    
    # Post-Exploitation (20-24)
    BLOODHOUND = 20
    POWERSHELL_EMPIRE = 21
    IMPACKET = 22
    CRACKMAPEXEC = 23
    ETERNALBLUE = 24


class DefensiveTool(Enum):
    """Real-World Defensive Tools (25 tools)"""
    # Network Defense (0-4)
    SNORT = 0
    SURICATA = 1
    ZEEK = 2
    WIRESHARK = 3
    TCPDUMP = 4
    
    # Firewall (5-9)
    IPTABLES = 5
    FIREWALLD = 6
    UFW = 7
    PFSENSE = 8
    FORTINET = 9
    
    # SIEM/Logging (10-14)
    OSSEC = 10
    WAZUH = 11
    SPLUNK = 12
    ELK_STACK = 13
    GRAYLOG = 14
    
    # Endpoint Protection (15-19)
    CLAMAV = 15
    FAIL2BAN = 16
    TRIPWIRE = 17
    AIDE = 18
    OSQUERY = 19
    
    # Incident Response (20-24)
    VOLATILITY = 20
    AUTOPSY = 21
    VELOCIRAPTOR = 22
    YARA = 23
    GHIDRA = 24


class ACEACv2Environment(gym.Env):
    """
    ACEAC v2.0 - Advanced Cyber Kill Chain Environment
    Features: Kill Chain phases, Real tools, MITRE ATT&CK mapping
    """
    
    metadata = {'render_modes': [], 'name': 'ACEAC-v2-KillChain'}
    
    def __init__(self, agent_role="red"):
        super().__init__()
        
        # Observation space: [kill_chain_phase(7), network_state(20), tool_status(25), defense_posture(10)]
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(62,), dtype=np.float32
        )
        
        # Action space: 25 offensive or 25 defensive tools
        self.action_space = spaces.Discrete(25)
        
        self.agent_role = agent_role  # "red" or "blue"
        self.max_steps = 100
        
        # Kill Chain state
        self.current_phase = CyberKillChainPhase.RECONNAISSANCE
        self.phase_progress = {phase: 0.0 for phase in CyberKillChainPhase}
        
        # Network state
        self.network_security = 0.8
        self.detection_level = 0.3
        self.compromised_hosts = 0
        self.data_exfiltrated = 0.0
        
        # Tool effectiveness tracking
        self.tool_usage = {i: 0 for i in range(25)}
        self.successful_actions = 0
        self.failed_actions = 0

        # Blue agent tracking for behavioral characterization
        self.attacks_detected = 0
        self.total_attacks = 0
        self.false_positives = 0
        self.defensive_actions = 0
        self.recent_actions = []

        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.current_step = 0
        self.current_phase = CyberKillChainPhase.RECONNAISSANCE
        self.phase_progress = {phase: 0.0 for phase in CyberKillChainPhase}
        
        self.network_security = 0.8
        self.detection_level = 0.3
        self.compromised_hosts = 0
        self.data_exfiltrated = 0.0
        
        self.tool_usage = {i: 0 for i in range(25)}
        self.successful_actions = 0
        self.failed_actions = 0

        # Reset Blue tracking metrics
        self.attacks_detected = 0
        self.total_attacks = 0
        self.false_positives = 0
        self.defensive_actions = 0
        self.recent_actions = []

        return self._get_observation(), {}
    
    def step(self, action):
        self.current_step += 1
        
        if isinstance(action, np.ndarray):
            action = int(action.item())
        else:
            action = int(action)
        
        # Track tool usage
        self.tool_usage[action] += 1
        
        # Execute action based on role
        if self.agent_role == "red":
            reward, success = self._execute_offensive_action(action)
        else:
            reward, success = self._execute_defensive_action(action)
        
        if success:
            self.successful_actions += 1
        else:
            self.failed_actions += 1
        
        # Check termination
        terminated = (
            self.current_step >= self.max_steps or
            self.network_security <= 0.1 or
            self.network_security >= 0.95 or
            self.current_phase == CyberKillChainPhase.ACTIONS_OBJECTIVES
        )
        
        info = {
            'current_phase': self.current_phase.name,
            'network_security': float(self.network_security),
            'detection_level': float(self.detection_level),
            'compromised_hosts': int(self.compromised_hosts),
            'success_rate': float(self.successful_actions / (self.successful_actions + self.failed_actions + 1)),
            'kill_chain_progress': float(sum(self.phase_progress.values()) / 7.0)
        }
        
        return self._get_observation(), float(reward), terminated, False, info
    
    def _execute_offensive_action(self, tool_id: int) -> Tuple[float, bool]:
        """Execute offensive tool action"""
        
        # Tool effectiveness by phase
        tool_phase_map = {
            # Reconnaissance tools (0-4) → RECONNAISSANCE phase
            0: CyberKillChainPhase.RECONNAISSANCE,
            1: CyberKillChainPhase.RECONNAISSANCE,
            2: CyberKillChainPhase.RECONNAISSANCE,
            3: CyberKillChainPhase.RECONNAISSANCE,
            4: CyberKillChainPhase.RECONNAISSANCE,
            
            # Weaponization tools (5-9) → WEAPONIZATION phase
            5: CyberKillChainPhase.WEAPONIZATION,
            6: CyberKillChainPhase.WEAPONIZATION,
            7: CyberKillChainPhase.WEAPONIZATION,
            8: CyberKillChainPhase.WEAPONIZATION,
            9: CyberKillChainPhase.WEAPONIZATION,
            
            # Exploitation tools (10-14) → EXPLOITATION phase
            10: CyberKillChainPhase.EXPLOITATION,
            11: CyberKillChainPhase.EXPLOITATION,
            12: CyberKillChainPhase.EXPLOITATION,
            13: CyberKillChainPhase.EXPLOITATION,
            14: CyberKillChainPhase.EXPLOITATION,
            
            # Password/Lateral movement (15-19) → INSTALLATION phase
            15: CyberKillChainPhase.INSTALLATION,
            16: CyberKillChainPhase.INSTALLATION,
            17: CyberKillChainPhase.INSTALLATION,
            18: CyberKillChainPhase.INSTALLATION,
            19: CyberKillChainPhase.INSTALLATION,
            
            # Post-exploitation (20-24) → C2 / Actions
            20: CyberKillChainPhase.COMMAND_CONTROL,
            21: CyberKillChainPhase.COMMAND_CONTROL,
            22: CyberKillChainPhase.ACTIONS_OBJECTIVES,
            23: CyberKillChainPhase.ACTIONS_OBJECTIVES,
            24: CyberKillChainPhase.ACTIONS_OBJECTIVES,
        }
        
        expected_phase = tool_phase_map.get(tool_id, self.current_phase)
        
        # Base attack power
        base_power = 0.08 + (tool_id % 5) * 0.02
        
        # Phase alignment bonus
        if expected_phase == self.current_phase:
            phase_bonus = 1.5
        else:
            phase_bonus = 0.7
        
        # Calculate success
        attack_power = base_power * phase_bonus
        success_prob = attack_power * (1.3 - self.network_security) * (1.0 - self.detection_level * 0.5)
        success = np.random.random() < success_prob
        
        if success:
            # Successful attack
            damage = attack_power * np.random.uniform(0.8, 1.2)
            self.network_security = max(0.0, self.network_security - damage)
            
            # Progress kill chain
            self.phase_progress[self.current_phase] += 0.2
            
            if self.phase_progress[self.current_phase] >= 1.0:
                if self.current_phase.value < 6:
                    self.current_phase = CyberKillChainPhase(self.current_phase.value + 1)
            
            # Additional effects
            if tool_id in [10, 11, 12, 13, 14]:  # Exploitation tools
                self.compromised_hosts += 1
            
            if tool_id in [22, 23, 24]:  # Data exfiltration tools
                self.data_exfiltrated += 0.1
            
            reward = 20.0 + attack_power * 30.0 + (self.current_phase.value * 5.0)
        else:
            # Failed attack
            self.detection_level = min(1.0, self.detection_level + 0.05)
            reward = -10.0
        
        # Stealth penalties
        noisy_tools = [10, 14, 24]  # Metasploit, Cobalt Strike, EternalBlue
        if tool_id in noisy_tools:
            reward -= 3.0
            self.detection_level = min(1.0, self.detection_level + 0.1)
        
        return reward, success
    
    def _execute_defensive_action(self, tool_id: int) -> Tuple[float, bool]:
        """DENSE REWARD DEFENSIVE ACTION with detection, FP, response time, diversity"""

        # Track defensive action
        self.defensive_actions += 1
        self.recent_actions.append(tool_id)
        if len(self.recent_actions) > 25:
            self.recent_actions.pop(0)

        # Tool effectiveness by category (BALANCED at 0.20-0.30, 2.5× original)
        tool_effectiveness = {
            # Network monitoring (0-4)
            **{i: 0.20 for i in range(5)},
            # Firewall (5-9)
            **{i: 0.30 for i in range(5, 10)},
            # SIEM (10-14)
            **{i: 0.25 for i in range(10, 15)},
            # Endpoint (15-19)
            **{i: 0.22 for i in range(15, 20)},
            # Incident Response (20-24)
            **{i: 0.28 for i in range(20, 25)},
        }

        defense_power = tool_effectiveness.get(tool_id, 0.20)

        # Simulate incoming attack (70% chance attack is occurring)
        attack_occurring = np.random.random() < 0.7
        attack_intensity = np.random.uniform(0.3, 0.8)

        # Calculate defense effectiveness
        defense_effectiveness = defense_power * np.random.uniform(0.9, 1.3)
        response_time = np.random.uniform(0.0, 10.0)  # Response delay in arbitrary units

        # Detection logic
        detection_threshold = attack_intensity * 0.6
        detected = defense_effectiveness > detection_threshold
        raised_alarm = detected  # Simplified: assume alarm is raised when detection occurs

        # Initialize reward components
        detection_reward = 0.0
        time_bonus = 0.0

        if attack_occurring:
            self.total_attacks += 1

            if detected:
                # Successfully caught the attack!
                self.attacks_detected += 1
                detection_reward = +10.0
                time_bonus = 5.0 * (1.0 - response_time / 10.0)  # Faster response = higher bonus

                # Heal network security
                healing = defense_effectiveness * 0.05
                self.network_security = min(1.0, self.network_security + healing)
                self.detection_level = min(1.0, self.detection_level + 0.03)

                success = True
            else:
                # Missed the attack (false negative)
                detection_reward = -10.0
                success = False
        else:
            # No attack occurring
            if raised_alarm:
                # False positive: raised alarm when there was no attack
                self.false_positives += 1
                detection_reward = -5.0
                success = False
            else:
                # True negative: correctly didn't raise alarm
                detection_reward = +2.0
                success = True

        # Diversity bonus: reward using variety of tools
        unique_tools = len(set(self.recent_actions))
        diversity_bonus = 5.0 * (unique_tools / 25.0)

        # Total reward
        reward = detection_reward + time_bonus + diversity_bonus

        # Cost penalties for expensive tools
        expensive_tools = [12, 14, 8, 9]  # Splunk, Graylog, pfSense, Fortinet
        if tool_id in expensive_tools:
            reward -= 2.0

        return reward, success
    
    def _get_observation(self):
        """
        Observation: [kill_chain_phase(7), network_state(20), tool_status(25), defense_posture(10)]
        Total: 62 dimensions
        """
        obs = np.zeros(62, dtype=np.float32)
        
        # Kill chain phase (one-hot encoding, 7 dims)
        obs[self.current_phase.value] = 1.0
        
        # Network state (20 dims)
        obs[7] = self.network_security
        obs[8] = self.detection_level
        obs[9] = self.compromised_hosts / 10.0
        obs[10] = self.data_exfiltrated
        obs[11] = self.successful_actions / (self.successful_actions + self.failed_actions + 1)
        obs[12] = self.current_step / self.max_steps
        obs[13:27] = np.random.random(14) * 0.3
        
        # Tool usage status (25 dims)
        max_usage = max(self.tool_usage.values()) if self.tool_usage else 1
        for i in range(25):
            obs[27 + i] = self.tool_usage.get(i, 0) / (max_usage + 1)
        
        # Defense posture (10 dims)
        obs[52] = sum(self.phase_progress.values()) / 7.0
        obs[53:62] = np.random.random(9) * 0.4
        
        return obs


if __name__ == "__main__":
    print("\n" + "="*70)
    print("ACEAC v2.0 - Cyber Kill Chain + Real Tools Environment")
    print("="*70)
    print("User: sarowarzahan414")
    print("Date: 2025-10-08 23:14:00 UTC")
    print("="*70)
    print("\nFeatures:")
    print("  - 7 Cyber Kill Chain Phases")
    print("  - 25 Real Offensive Tools")
    print("  - 25 Real Defensive Tools")
    print("  - 62-dimensional observation space")
    print("  - Phase-aware action evaluation")
    print("="*70)
    
    # Test Red Agent
    print("\nTesting Red Agent Environment...")
    env_red = ACEACv2Environment(agent_role="red")
    obs, _ = env_red.reset()
    
    print(f"Observation shape: {obs.shape}")
    print(f"Action space: {env_red.action_space.n} tools")
    
    for i in range(5):
        action = env_red.action_space.sample()
        obs, reward, terminated, truncated, info = env_red.step(action)
        print(f"Step {i+1}: Tool={action}, Phase={info['current_phase']}, "
              f"Reward={reward:.1f}, Security={info['network_security']:.2f}")
        if terminated:
            break
    
    print("\n" + "="*70)
    print("Environment test complete!")
    print("="*70 + "\n")
