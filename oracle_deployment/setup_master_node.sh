#!/bin/bash

################################################################################
# ACEAC Master Node Setup for Oracle Free Tier
# Installs Python, RL libraries, and offensive security tools
#
# Author: @sarowarzahan414
# Date: 2024-12-24
################################################################################

set -e  # Exit on error

echo "========================================================================"
echo "ACEAC Master Node Setup - Oracle Free Tier"
echo "========================================================================"
echo "This will install:"
echo "  - Python 3.10 + ML libraries"
echo "  - Offensive security tools (Nmap, Masscan, SQLMap, etc.)"
echo "  - ACEAC framework"
echo "========================================================================"
echo ""

# Update system
echo "[1/6] Updating system packages..."
sudo apt update
sudo apt upgrade -y

# Install Python and dependencies
echo "[2/6] Installing Python 3.10..."
sudo apt install -y \
    python3.10 \
    python3-pip \
    python3-venv \
    python3-dev \
    build-essential \
    git \
    curl \
    wget \
    htop \
    tmux

# Create Python virtual environment
echo "[3/6] Creating Python virtual environment..."
cd ~
python3 -m venv aceac_env
source aceac_env/bin/activate

# Install Python packages
echo "[4/6] Installing Python packages..."
pip install --upgrade pip
pip install \
    gymnasium \
    stable-baselines3 \
    numpy \
    scipy \
    matplotlib \
    pandas \
    pyyaml \
    requests

# Install offensive security tools
echo "[5/6] Installing offensive security tools..."

# Network scanning
sudo apt install -y nmap
sudo apt install -y masscan

# Web application testing
sudo apt install -y sqlmap
sudo apt install -y nikto

# Password cracking
sudo apt install -y hydra
sudo apt install -y john
sudo apt install -y hashcat

# General tools
sudo apt install -y netcat
sudo apt install -y curl
sudo apt install -y whois
sudo apt install -y dnsutils

# Install Metasploit Framework (optional, large download)
echo "Do you want to install Metasploit Framework? (y/n)"
read -r install_msf
if [ "$install_msf" = "y" ]; then
    echo "Installing Metasploit Framework..."
    curl https://raw.githubusercontent.com/rapid7/metasploit-omnibus/master/config/templates/metasploit-framework-wrappers/msfupdate.erb > msfinstall
    chmod 755 msfinstall
    sudo ./msfinstall
    rm msfinstall
fi

# Clone ACEAC repository
echo "[6/6] Setting up ACEAC..."
cd ~
if [ ! -d "ACEAC-local" ]; then
    git clone https://github.com/sarowarzahan414/ACEAC-local.git
fi

cd ACEAC-local

# Create config file
cat > config.yaml <<EOF
# ACEAC Configuration for Oracle Free Tier

# Network Configuration
network:
  master_ip: "10.0.1.10"
  target_ip: "10.0.1.100"
  defense_ip: "10.0.1.20"
  analytics_ip: "10.0.1.30"
  subnet: "10.0.1.0/24"

# Authorized Targets (ONLY these IPs can be tested)
authorized_targets:
  - "10.0.1.0/24"
  - "127.0.0.1"

# QD Pool Configuration
qd_pool:
  grid_resolution: 10
  behavior_dimensions: 2
  generations: 10
  episodes_per_generation: 100

# Tool Configuration
tools:
  rate_limits:
    nmap:
      per_minute: 2
      per_hour: 20
    masscan:
      per_minute: 1
      per_hour: 10
    sqlmap:
      per_minute: 1
      per_hour: 10
    hydra:
      per_minute: 5
      per_hour: 50

# Logging
logging:
  level: "INFO"
  audit_log: "/var/log/aceac/audit.log"
  results_dir: "/var/log/aceac/results"

# Oracle Free Tier Optimization
oracle:
  compute_shape: "VM.Standard.A1.Flex"
  ocpus: 2
  memory_gb: 12
EOF

# Create log directories
sudo mkdir -p /var/log/aceac/results
sudo chown -R $USER:$USER /var/log/aceac

# Create requirements file
cat > requirements_real_world.txt <<EOF
gymnasium>=0.29.0
stable-baselines3>=2.1.0
numpy>=1.24.0
scipy>=1.11.0
matplotlib>=3.7.0
pandas>=2.0.0
pyyaml>=6.0
requests>=2.31.0
EOF

pip install -r requirements_real_world.txt

echo ""
echo "========================================================================"
echo "Master Node Setup Complete!"
echo "========================================================================"
echo ""
echo "Next steps:"
echo "  1. Activate virtual environment: source ~/aceac_env/bin/activate"
echo "  2. Configure targets in config.yaml"
echo "  3. Set up target VMs (run setup_target_node.sh on target VM)"
echo "  4. Run: python aceac_qd_pool_real_world.py"
echo ""
echo "To test tools:"
echo "  nmap --version"
echo "  masscan --version"
echo "  sqlmap --version"
echo ""
echo "========================================================================"
