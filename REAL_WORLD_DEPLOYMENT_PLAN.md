# ACEAC Real-World Deployment Architecture
## QD Pool + Behavioral Characterization on Oracle Free Tier

**Author**: @sarowarzahan414
**Date**: 2024-12-24
**Innovation**: Real-world QD-based Cyber Range Testing

---

## Executive Summary

Deploy ACEAC with Quality Diversity (QD) Pool and Behavioral Characterization to **actual cloud infrastructure** where AI agents execute **real security tools** against **real systems** in a controlled, ethical, sandboxed environment.

**Why This Matters**:
- Current research uses simulated environments (abstract reward functions)
- Real-world deployment tests actual tool effectiveness
- QD diversity ensures comprehensive security testing
- Oracle Free Tier makes it accessible to researchers worldwide

---

## Current State vs Real-World

### Current Implementation (Simulated)
```python
# Abstract simulation
attack_power = 0.08 + action * 0.015
self.network_security -= attack_power
reward = attack_power * 100
```

### Real-World Implementation (Proposed)
```python
# Execute actual tool
result = execute_tool(OffensiveTool.NMAP, target_host)
vulnerabilities_found = parse_nmap_output(result)
reward = calculate_reward(vulnerabilities_found)
```

---

## Architecture Overview

### 1. Infrastructure Layout (Oracle Free Tier)

**Oracle Free Tier Resources** (Always Free):
- 2 AMD-based Compute VMs (1/8 OCPU, 1 GB RAM each) OR
- 4 Arm-based Ampere A1 cores (24 GB RAM total)
- 200 GB Block Volume storage
- 10 TB/month egress

**Proposed Setup**:
```
┌─────────────────────────────────────────────────────┐
│              Oracle Cloud Free Tier                 │
├─────────────────────────────────────────────────────┤
│                                                      │
│  ┌────────────────┐      ┌────────────────┐        │
│  │   VM1: Master  │      │  VM2: Target   │        │
│  │  - ACEAC Engine│◄────►│  - Vulnerable  │        │
│  │  - Red Agent   │      │    Services    │        │
│  │  - Blue Agent  │      │  - Monitoring  │        │
│  │  - QD Pool     │      └────────────────┘        │
│  └────────────────┘                                 │
│         │                                            │
│         ▼                                            │
│  ┌────────────────┐      ┌────────────────┐        │
│  │  VM3: Defense  │      │ VM4: Analytics │        │
│  │  - SIEM/WAZUH  │      │  - Results DB  │        │
│  │  - IDS/Snort   │      │  - Metrics     │        │
│  │  - Honeypot    │      │  - Dashboards  │        │
│  └────────────────┘      └────────────────┘        │
│                                                      │
└─────────────────────────────────────────────────────┘
```

### 2. Tool Execution Bridge

**Tool Wrapper System**:
```python
class RealToolExecutor:
    """Execute actual security tools with safety controls"""

    ALLOWED_TARGETS = ['192.168.1.0/24']  # Sandboxed network only
    RATE_LIMITS = {
        'nmap': 1,      # 1 scan per minute
        'metasploit': 5, # 5 exploits per hour
        'hydra': 10     # 10 attempts per minute
    }

    def execute(self, tool: OffensiveTool, target: str, params: dict):
        # Safety checks
        if not self.is_authorized_target(target):
            raise SecurityError("Unauthorized target")

        if not self.check_rate_limit(tool):
            raise RateLimitError("Tool rate limit exceeded")

        # Execute real tool
        if tool == OffensiveTool.NMAP:
            return self.run_nmap(target, params)
        elif tool == OffensiveTool.METASPLOIT:
            return self.run_metasploit(target, params)
        # ... etc
```

### 3. Real Tool Implementations

**Example: Nmap Integration**
```python
def run_nmap(self, target: str, params: dict) -> ToolResult:
    """Execute real nmap scan"""
    cmd = [
        'nmap',
        '-sV',  # Version detection
        '-T4',  # Timing
        '--max-retries', '2',
        target
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        timeout=300,  # 5 min timeout
        check=False
    )

    return ToolResult(
        success=result.returncode == 0,
        output=result.stdout.decode(),
        ports_found=self.parse_nmap_ports(result.stdout),
        vulnerabilities=self.parse_nmap_vulns(result.stdout)
    )
```

**Example: Defensive Tool - Snort IDS**
```python
def run_snort(self, interface: str, duration: int) -> ToolResult:
    """Monitor network traffic with Snort"""
    cmd = [
        'snort',
        '-i', interface,
        '-c', '/etc/snort/snort.conf',
        '-l', '/var/log/snort',
        '-A', 'console'
    ]

    # Run for specified duration
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    time.sleep(duration)
    proc.terminate()

    alerts = self.parse_snort_alerts('/var/log/snort')

    return ToolResult(
        success=True,
        alerts_detected=len(alerts),
        alert_types=self.categorize_alerts(alerts)
    )
```

---

## QD Pool for Real-World Testing

### Why QD Matters in Real Environments

**Problem with Standard RL**:
- Converges to single strategy (e.g., always uses Metasploit)
- Misses diverse attack vectors
- Real defenders can easily detect patterns

**QD Pool Solution**:
- Maintains diverse attack/defense strategies
- Maps behavioral space (stealth vs aggressive, fast vs thorough)
- Discovers novel tool combinations

### Behavioral Characterization for Real Tools

**Behavior Dimensions**:
1. **Stealth Score**: Detection rate by IDS/SIEM
2. **Tool Diversity**: Number of unique tools used
3. **Kill Chain Progress**: Which phases reached
4. **Resource Efficiency**: CPU/network usage

```python
def compute_real_behavior(episode_log: EpisodeLog) -> np.ndarray:
    """Compute behavioral characterization from real execution"""

    # Dimension 1: Stealth (0 = detected, 1 = undetected)
    stealth = 1.0 - (episode_log.ids_alerts / max(episode_log.actions, 1))

    # Dimension 2: Tool diversity (0-1)
    unique_tools = len(set(episode_log.tools_used))
    diversity = unique_tools / 25.0  # 25 total tools

    # Dimension 3: Kill chain depth (0-1)
    max_phase = max(episode_log.phases_reached)
    kill_chain = max_phase / 6.0  # 7 phases (0-6)

    # Dimension 4: Speed (0 = slow, 1 = fast)
    speed = 1.0 - (episode_log.duration / 600.0)  # 10 min max

    return np.array([stealth, diversity, kill_chain, speed])
```

---

## Safety & Ethics Controls

### 1. Network Isolation
```bash
# Oracle Cloud VCN (Virtual Cloud Network) isolation
# All testing contained within private subnet
# No external internet access for target VMs
# Firewall rules restrict all outbound traffic
```

### 2. Authorization System
```python
class AuthorizationManager:
    """Ensure all actions are authorized"""

    def __init__(self):
        self.authorized_targets = set()
        self.authorized_users = set()
        self.session_token = None

    def require_authorization(self, action: str):
        """Decorator requiring explicit authorization"""
        if not self.is_authorized_session():
            raise AuthorizationError(
                "This action requires explicit authorization.\n"
                "This system is for AUTHORIZED TESTING ONLY.\n"
                "Unauthorized access is prohibited by law."
            )
```

### 3. Audit Logging
```python
# Every action logged
audit_log = {
    'timestamp': datetime.now(timezone.utc).isoformat(),
    'user': 'sarowarzahan414',
    'action': 'nmap_scan',
    'target': '192.168.1.100',
    'authorization': 'research-session-2024',
    'result': 'success',
    'findings': {...}
}
```

---

## Oracle Free Tier Deployment Guide

### Step 1: Create VMs

```bash
# Use Oracle Cloud CLI or Web Console
# Create 4 Ampere A1 instances (6 GB RAM each)

# VM1: Master (Ubuntu 22.04)
Name: aceac-master
Shape: VM.Standard.A1.Flex (2 OCPUs, 12 GB RAM)
OS: Ubuntu 22.04 LTS
Purpose: ACEAC engine, agents, QD pool

# VM2: Target (Ubuntu 22.04)
Name: aceac-target
Shape: VM.Standard.A1.Flex (1 OCPU, 6 GB RAM)
OS: Ubuntu 22.04 LTS
Purpose: Vulnerable services, monitoring

# VM3: Defense (Ubuntu 22.04)
Name: aceac-defense
Shape: VM.Standard.A1.Flex (1 OCPU, 6 GB RAM)
OS: Ubuntu 22.04 LTS
Purpose: SIEM, IDS, defensive tools

# VM4: Analytics (Ubuntu 22.04)
Name: aceac-analytics
Shape: VM.Standard.A1.Flex (0.5 OCPU, 3 GB RAM)
OS: Ubuntu 22.04 LTS
Purpose: Results database, metrics
```

### Step 2: Network Configuration

```bash
# Create isolated VCN
VCN CIDR: 10.0.0.0/16
Subnet: 10.0.1.0/24 (private)

# Security List Rules (DENY ALL by default)
Ingress:
  - Allow 10.0.1.0/24 → 10.0.1.0/24 (all ports)
  - Allow SSH from your IP (port 22)

Egress:
  - Allow 10.0.1.0/24 → 10.0.1.0/24 (all ports)
  - DENY all internet access (except package repos)
```

### Step 3: Install Tools

**Master Node**:
```bash
# Python environment
sudo apt update
sudo apt install -y python3.10 python3-pip
pip3 install gymnasium stable-baselines3 numpy

# Security tools (offensive)
sudo apt install -y nmap masscan sqlmap hydra john
git clone https://github.com/rapid7/metasploit-framework
```

**Defense Node**:
```bash
# IDS/IPS
sudo apt install -y snort suricata

# SIEM
curl -s https://packages.wazuh.com/key/GPG-KEY-WAZUH | sudo apt-key add -
echo "deb https://packages.wazuh.com/4.x/apt/ stable main" | sudo tee /etc/apt/sources.list.d/wazuh.list
sudo apt update
sudo apt install -y wazuh-manager
```

**Target Node**:
```bash
# Intentionally vulnerable services (for testing)
# DVWA (Damn Vulnerable Web Application)
git clone https://github.com/digininja/DVWA.git
docker-compose up -d

# Metasploitable-like services
sudo apt install -y vsftpd telnetd
```

### Step 4: Deploy ACEAC

```bash
# On master node
git clone https://github.com/sarowarzahan414/ACEAC-local.git
cd ACEAC-local

# Install with real-world extensions
pip3 install -r requirements_real_world.txt

# Configure
cp config.example.yaml config.yaml
# Edit config.yaml with your Oracle IPs
```

---

## Cost Analysis

**Oracle Free Tier (Always Free)**:
- Compute: $0 (4 Ampere A1 cores, 24 GB RAM)
- Storage: $0 (200 GB Block Volume)
- Network: $0 (10 TB egress/month)
- **Total: $0/month FOREVER**

**Comparison**:
- AWS equivalent: ~$150/month
- Azure equivalent: ~$180/month
- GCP equivalent: ~$140/month

**Oracle Free Tier is PERFECT for research!**

---

## Implementation Timeline

1. **Week 1**: Oracle infrastructure setup
2. **Week 2**: Tool integration + safety controls
3. **Week 3**: QD Pool implementation
4. **Week 4**: Real-world testing + validation
5. **Week 5**: Results analysis + paper writing

---

## Expected Research Outcomes

### Publications
- "Quality Diversity for Real-World Cyber Defense" (ACM CCS)
- "Bridging Simulation to Reality in Adversarial ML" (USENIX Security)
- "Behavioral Characterization of Security Tools" (NDSS)

### Metrics to Collect
- Real tool effectiveness vs simulation accuracy
- QD diversity in real environments
- Novel attack/defense strategies discovered
- Computational cost vs effectiveness

### Open Source Release
- ACEAC Real-World Extension (GitHub)
- Oracle Free Tier deployment scripts
- Pre-trained QD pools
- Research dataset (anonymized)

---

## Next Steps

1. ✅ You approve this architecture
2. Create Oracle Free Tier account
3. Deploy infrastructure (automated scripts)
4. Run first real-world QD experiment
5. Compare real vs simulated results
6. Publish findings

**Ready to proceed?**
