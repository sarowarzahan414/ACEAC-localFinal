## ACEAC Real-World Deployment - Oracle Free Tier Setup Guide

**Complete step-by-step guide to deploy ACEAC on Oracle Cloud Free Tier**

**Author**: @sarowarzahan414
**Date**: 2024-12-24
**Innovation**: Real-world QD Pool testing on $0/month infrastructure

---

## Overview

This guide will help you deploy a complete ACEAC real-world testing environment using Oracle's Always Free tier. You'll create 4 VMs running offensive tools, vulnerable targets, defensive tools, and analytics - all at **zero cost**.

---

## Prerequisites

1. Oracle Cloud account (create at https://cloud.oracle.com/free)
2. SSH key pair
3. Basic Linux knowledge
4. ~2 hours of setup time

---

## Part 1: Oracle Cloud Setup

### Step 1: Create Oracle Cloud Account

1. Go to https://cloud.oracle.com/free
2. Click "Start for free"
3. Fill in details (requires credit card for verification, but won't charge)
4. Verify email
5. Log in to Oracle Cloud Console

### Step 2: Create VCN (Virtual Cloud Network)

1. Navigate to: **Networking → Virtual Cloud Networks**
2. Click **Create VCN**
3. Fill in:
   - **Name**: `aceac-vcn`
   - **CIDR Block**: `10.0.0.0/16`
   - **DNS Label**: `aceacvcn`
4. Click **Create VCN**

### Step 3: Create Subnet

1. In your VCN, click **Create Subnet**
2. Fill in:
   - **Name**: `aceac-private-subnet`
   - **CIDR Block**: `10.0.1.0/24`
   - **Subnet Type**: Regional
   - **Subnet Access**: Private
3. Click **Create Subnet**

### Step 4: Configure Security List

1. Navigate to **Security Lists** in your VCN
2. Click **Default Security List**
3. Add **Ingress Rules**:

```
Type: All Traffic
Source: 10.0.1.0/24
Description: Allow internal traffic

Type: SSH
Source: 0.0.0.0/0 (or your IP)
Port: 22
Description: SSH access
```

4. Add **Egress Rules**:

```
Type: All Traffic
Destination: 10.0.1.0/24
Description: Allow internal traffic

Type: HTTP/HTTPS
Destination: 0.0.0.0/0
Description: Package downloads (can remove after setup)
```

### Step 5: Create Compute Instances

**Oracle Free Tier Ampere A1 allocation:**
- Total: 4 OCPUs, 24 GB RAM (Always Free)
- Our distribution: 4 VMs

#### VM1: Master Node (ACEAC Engine)

1. Navigate to: **Compute → Instances**
2. Click **Create Instance**
3. Fill in:
   - **Name**: `aceac-master`
   - **Image**: Ubuntu 22.04 LTS
   - **Shape**: VM.Standard.A1.Flex
   - **OCPUs**: 2
   - **Memory**: 12 GB
   - **VCN**: aceac-vcn
   - **Subnet**: aceac-private-subnet
   - **Private IP**: 10.0.1.10 (static)
   - **Public IP**: Assign (for SSH access)
   - **SSH Keys**: Upload your public key
4. Click **Create**

#### VM2: Target Node (Vulnerable Services)

Repeat above with:
- **Name**: `aceac-target`
- **OCPUs**: 1
- **Memory**: 6 GB
- **Private IP**: 10.0.1.100
- **Public IP**: None (private only for safety)

#### VM3: Defense Node (IDS/SIEM)

Repeat with:
- **Name**: `aceac-defense`
- **OCPUs**: 1
- **Memory**: 6 GB
- **Private IP**: 10.0.1.20
- **Public IP**: None

#### VM4: Analytics Node (Results DB)

Repeat with:
- **Name**: `aceac-analytics`
- **OCPUs**: 0.5 (Flexible)
- **Memory**: 3 GB
- **Private IP**: 10.0.1.30
- **Public IP**: None

**Note**: You may need to create VMs one at a time if you hit capacity limits. Oracle allows requests for more Always Free resources.

---

## Part 2: VM Configuration

### Connect to Master Node

```bash
# From your local machine
ssh -i ~/.ssh/your_key ubuntu@<master_public_ip>
```

### Setup Master Node

```bash
# Download setup script
wget https://raw.githubusercontent.com/sarowarzahan414/ACEAC-local/main/oracle_deployment/setup_master_node.sh

# Make executable
chmod +x setup_master_node.sh

# Run setup (takes ~15 minutes)
./setup_master_node.sh
```

The script will install:
- Python 3.10 + ML libraries (Gymnasium, Stable-Baselines3)
- Offensive tools (Nmap, Masscan, SQLMap, Hydra)
- ACEAC framework
- Configuration files

### Setup Target Node

```bash
# SSH to target node FROM master node (using private IP)
ssh ubuntu@10.0.1.100

# Download setup script
wget https://raw.githubusercontent.com/sarowarzahan414/ACEAC-local/main/oracle_deployment/setup_target_node.sh

chmod +x setup_target_node.sh

# WARNING: This creates intentionally vulnerable services
# Only run in isolated environment
./setup_target_node.sh
```

This installs:
- DVWA (Damn Vulnerable Web App)
- WebGoat
- Mutillidae
- Vulnerable FTP, SSH, MySQL
- Test users with weak passwords

### Setup Defense Node

```bash
# SSH to defense node
ssh ubuntu@10.0.1.20

# Download and run setup
wget https://raw.githubusercontent.com/sarowarzahan414/ACEAC-local/main/oracle_deployment/setup_defense_node.sh

chmod +x setup_defense_node.sh
./setup_defense_node.sh
```

This installs:
- Snort IDS
- Suricata IDS
- Wazuh SIEM
- Monitoring tools

---

## Part 3: Running ACEAC

### Test Connectivity

From master node, verify access to other nodes:

```bash
# Test target
ping -c 3 10.0.1.100

# Test defense
ping -c 3 10.0.1.20

# Test analytics
ping -c 3 10.0.1.30
```

### Activate Virtual Environment

```bash
cd ~/ACEAC-local
source ~/aceac_env/bin/activate
```

### Run Simulated Training (Fast)

```bash
# Train QD Pool in simulation mode (no real tools)
python aceac_qd_pool_real_world.py

# Takes ~7 minutes for 10 generations, 100 episodes each
```

### Run Real-World Testing (Slow, Real Tools)

```bash
# Edit the script to enable real tool execution
nano aceac_qd_pool_real_world.py

# Change at the bottom:
# use_real_tools=True  # Enable real tools

# Then run
python aceac_qd_pool_real_world.py
```

This will:
1. Execute real nmap/masscan scans against target
2. Log all tool executions
3. Track IDS/SIEM alerts from defense node
4. Compute behavioral diversity based on real metrics
5. Build QD pool of diverse attack/defense strategies

### Monitor in Real-Time

On defense node, monitor attacks:

```bash
# Real-time monitoring dashboard
./monitor_attacks.sh
```

You'll see:
- Snort alerts
- Suricata alerts
- Wazuh SIEM alerts
- Active connections
- Top talkers

---

## Part 4: Results Analysis

### View Training Results

```bash
cd ~/ACEAC-local/logs

# View QD Pool results
cat qd_pool_real_world.json | jq .

# Key metrics
cat qd_pool_real_world.json | jq '.red_pool_final'
cat qd_pool_real_world.json | jq '.blue_pool_final'
```

### Analyze Behavioral Diversity

```python
import json
import matplotlib.pyplot as plt
import numpy as np

# Load results
with open('logs/qd_pool_real_world.json') as f:
    data = json.load(f)

# Plot coverage over generations
gens = [h['generation'] for h in data['history']]
red_cov = [h['red_pool']['coverage'] * 100 for h in data['history']]
blue_cov = [h['blue_pool']['coverage'] * 100 for h in data['history']]

plt.figure(figsize=(10, 6))
plt.plot(gens, red_cov, 'r-o', label='Red Team Coverage')
plt.plot(gens, blue_cov, 'b-o', label='Blue Team Coverage')
plt.xlabel('Generation')
plt.ylabel('Behavioral Coverage (%)')
plt.title('QD Pool Diversity Evolution')
plt.legend()
plt.grid(True)
plt.savefig('qd_coverage.png', dpi=300)
plt.show()
```

### Compare Simulation vs Real-World

Run both modes and compare:

```bash
# Simulated
python aceac_qd_pool_real_world.py  # use_real_tools=False
mv logs/qd_pool_real_world.json logs/qd_simulated.json

# Real-world
# (Edit script: use_real_tools=True)
python aceac_qd_pool_real_world.py
mv logs/qd_pool_real_world.json logs/qd_real_world.json

# Compare
python compare_results.py logs/qd_simulated.json logs/qd_real_world.json
```

---

## Part 5: Cost Analysis

**Oracle Free Tier (Always Free)**:
- Compute: 4 Ampere A1 cores, 24 GB RAM: **$0**
- Storage: 200 GB Block Volume: **$0**
- Network: 10 TB egress/month: **$0**
- **Total: $0/month FOREVER**

**Equivalent on other clouds**:
- AWS: ~$150/month
- Azure: ~$180/month
- GCP: ~$140/month

**Savings over 1 year**: ~$1,800

---

## Part 6: Safety & Ethics

### Authorization System

ACEAC includes built-in authorization:

```python
from aceac_real_world_bridge import RealToolExecutor

executor = RealToolExecutor()

# MUST authorize before use
session_id = executor.auth.authorize_session(
    user='sarowarzahan414',
    purpose='Security Research - QD Pool Development'
)

# All actions logged
# Unauthorized targets blocked
# Rate limiting enforced
```

### Audit Logs

All tool executions logged to:
- `/var/log/aceac/audit.log`
- Includes: timestamp, user, action, target, result

### Network Isolation

- All testing confined to `10.0.1.0/24`
- No external internet access for target/defense VMs
- Firewall rules restrict all outbound traffic
- Oracle Security Lists enforce isolation

### Legal Compliance

**IMPORTANT**: This system is for:
✅ Authorized security research
✅ Penetration testing with permission
✅ Educational purposes
✅ CTF competitions
✅ Defensive security

**NOT for**:
❌ Unauthorized access to systems
❌ Real-world attacks
❌ Testing systems you don't own
❌ Any illegal activity

---

## Part 7: Troubleshooting

### Common Issues

**"Permission denied" errors**:
```bash
# Ensure proper authorization
python -c "from aceac_real_world_bridge import RealToolExecutor; e=RealToolExecutor(); e.auth.authorize_session('your_user', 'purpose')"
```

**Tools not found**:
```bash
# Verify installation
which nmap
which masscan
which sqlmap

# Reinstall if needed
sudo apt install -y nmap masscan sqlmap
```

**Out of memory on small VMs**:
```bash
# Create swap file
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

**IDS not detecting attacks**:
```bash
# Check if Snort is running
sudo systemctl status snort

# Restart if needed
sudo systemctl restart snort

# Verify rules are loaded
sudo snort -T -c /etc/snort/snort.conf
```

---

## Part 8: Next Steps

### Research Opportunities

1. **Sim-to-Real Transfer**
   - Train in simulation
   - Validate on real tools
   - Measure transfer gap

2. **Novel Tool Combinations**
   - Let QD discover new attack sequences
   - Analyze what works in reality vs simulation

3. **Behavioral Diversity**
   - Map behavioral space of real attacks
   - Compare diversity vs standard RL

4. **Publications**
   - "Quality Diversity for Real-World Cyber Defense"
   - "Bridging Simulation to Reality in Adversarial ML"
   - "Oracle Free Tier for Security Research"

### Extend the System

1. **Add More Tools**
   - Implement remaining 50 tools
   - Add custom tools
   - Integrate with Metasploit

2. **Multi-Agent**
   - Multiple red agents collaborating
   - Team-based blue defense
   - Swarm attacks

3. **Advanced Behaviors**
   - APT-style multi-stage attacks
   - Stealth optimization
   - Time-based attacks

4. **Visualization**
   - Real-time attack dashboards
   - Behavioral heatmaps
   - Tool effectiveness matrices

---

## Part 9: Cleanup

### Stop All Services

```bash
# On each node
sudo systemctl stop snort
sudo systemctl stop suricata
sudo systemctl stop wazuh-manager
sudo docker stop $(sudo docker ps -aq)
```

### Terminate VMs (if done)

1. Oracle Cloud Console
2. Navigate to **Compute → Instances**
3. Select each VM
4. **More Actions → Terminate**
5. Confirm

**Note**: Free tier VMs can be kept running at no cost

---

## Conclusion

You now have a complete real-world cyber range running on Oracle Free Tier with:

✅ **Offensive Tools**: Nmap, Masscan, SQLMap, Hydra, etc.
✅ **Vulnerable Targets**: DVWA, WebGoat, weak services
✅ **Defensive Tools**: Snort, Suricata, Wazuh
✅ **QD Pool**: Behavioral diversity with MAP-Elites
✅ **Real Execution**: Actual tool execution, not simulation
✅ **Zero Cost**: $0/month forever

**This is a research game-changer!**

Anyone can now replicate real-world security AI research without expensive infrastructure.

---

## Support & Contributions

- **Issues**: https://github.com/sarowarzahan414/ACEAC-local/issues
- **Pull Requests**: Welcome!
- **Discussions**: Use GitHub Discussions

---

**Happy hacking (ethically)!** 🔒🤖

**@sarowarzahan414**
