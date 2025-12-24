#!/bin/bash

################################################################################
# ACEAC Defense Node Setup for Oracle Free Tier
# Installs SIEM, IDS, and monitoring tools
#
# Author: @sarowarzahan414
# Date: 2024-12-24
################################################################################

set -e

echo "========================================================================"
echo "ACEAC Defense Node Setup - Oracle Free Tier"
echo "========================================================================"
echo "This will install:"
echo "  - Snort IDS"
echo "  - Suricata IDS"
echo "  - Wazuh SIEM"
echo "  - Monitoring tools"
echo "========================================================================"
echo ""

# Update system
echo "[1/5] Updating system..."
sudo apt update
sudo apt upgrade -y

# Install dependencies
echo "[2/5] Installing dependencies..."
sudo apt install -y \
    build-essential \
    libpcap-dev \
    libpcre3-dev \
    libdumbnet-dev \
    bison \
    flex \
    zlib1g-dev \
    liblzma-dev \
    openssl \
    libssl-dev \
    pkg-config \
    curl \
    wget \
    git

# Install Snort IDS
echo "[3/5] Installing Snort IDS..."
sudo apt install -y snort

# Configure Snort
sudo mkdir -p /etc/snort/rules
sudo mkdir -p /var/log/snort

# Basic Snort rules for common attacks
cat | sudo tee /etc/snort/rules/local.rules <<EOF
# Port scanning detection
alert tcp any any -> \$HOME_NET any (msg:"SCAN nmap TCP"; flags:S; threshold: type both, track by_src, count 20, seconds 60; sid:1000001; rev:1;)

# SQL injection attempts
alert tcp any any -> \$HOME_NET 80 (msg:"WEB SQL injection attempt"; content:"union"; nocase; content:"select"; nocase; sid:1000002; rev:1;)

# Brute force SSH
alert tcp any any -> \$HOME_NET 22 (msg:"SCAN SSH brute force attempt"; threshold: type both, track by_src, count 10, seconds 60; sid:1000003; rev:1;)

# FTP anonymous login
alert tcp any any -> \$HOME_NET 21 (msg:"FTP anonymous login"; content:"USER anonymous"; sid:1000004; rev:1;)

# Metasploit activity
alert tcp any any -> \$HOME_NET any (msg:"EXPLOIT Metasploit activity detected"; content:"Metasploit"; sid:1000005; rev:1;)
EOF

# Configure Snort
sudo sed -i 's|# include \$RULE_PATH/local.rules|include \$RULE_PATH/local.rules|' /etc/snort/snort.conf

# Install Suricata IDS
echo "[4/5] Installing Suricata IDS..."
sudo apt install -y software-properties-common
sudo add-apt-repository -y ppa:oisf/suricata-stable
sudo apt update
sudo apt install -y suricata

# Configure Suricata
sudo suricata-update
sudo systemctl enable suricata
sudo systemctl start suricata

# Install Wazuh SIEM
echo "[5/5] Installing Wazuh SIEM..."

# Install dependencies
sudo apt install -y apt-transport-https lsb-release gnupg2

# Add Wazuh repository
curl -s https://packages.wazuh.com/key/GPG-KEY-WAZUH | sudo apt-key add -
echo "deb https://packages.wazuh.com/4.x/apt/ stable main" | sudo tee /etc/apt/sources.list.d/wazuh.list

sudo apt update
sudo apt install -y wazuh-manager

# Enable and start Wazuh
sudo systemctl enable wazuh-manager
sudo systemctl start wazuh-manager

# Install Wazuh agent (to monitor this node)
sudo apt install -y wazuh-agent

# Configure Wazuh agent
echo "WAZUH_MANAGER='127.0.0.1'" | sudo tee -a /var/ossec/etc/ossec.conf
sudo systemctl enable wazuh-agent
sudo systemctl start wazuh-agent

# Install monitoring tools
echo "Installing monitoring tools..."
sudo apt install -y \
    tcpdump \
    wireshark \
    tshark \
    ngrep \
    iftop \
    nethogs \
    iptraf-ng

# Create monitoring script
cat > ~/monitor_attacks.sh <<'EOF'
#!/bin/bash
# Real-time attack monitoring script

echo "========================================================================"
echo "ACEAC Real-Time Attack Monitor"
echo "========================================================================"
echo ""

while true; do
    clear
    echo "=== Snort Alerts (last 10) ==="
    sudo tail -10 /var/log/snort/alert 2>/dev/null || echo "No Snort alerts"
    echo ""

    echo "=== Suricata Alerts (last 5) ==="
    sudo tail -5 /var/log/suricata/fast.log 2>/dev/null || echo "No Suricata alerts"
    echo ""

    echo "=== Wazuh Alerts (last 5) ==="
    sudo tail -5 /var/ossec/logs/alerts/alerts.log 2>/dev/null || echo "No Wazuh alerts"
    echo ""

    echo "=== Active Connections ==="
    sudo netstat -tn | grep ESTABLISHED | wc -l
    echo "connections active"
    echo ""

    echo "=== Top Talkers ==="
    sudo tcpdump -i any -c 100 -nn 2>/dev/null | awk '{print $3}' | sort | uniq -c | sort -rn | head -5
    echo ""

    echo "Press Ctrl+C to exit"
    sleep 5
done
EOF

chmod +x ~/monitor_attacks.sh

# Create log analysis script
cat > ~/analyze_logs.py <<'EOF'
#!/usr/bin/env python3
"""
ACEAC Log Analysis Script
Analyzes IDS/SIEM logs and provides metrics
"""

import json
import re
from collections import Counter
from datetime import datetime

def analyze_snort_logs(log_file='/var/log/snort/alert'):
    """Analyze Snort alerts"""
    try:
        with open(log_file, 'r') as f:
            alerts = f.readlines()

        # Count alert types
        alert_types = Counter()
        for alert in alerts:
            if 'msg:' in alert:
                msg = re.search(r'msg:"([^"]+)"', alert)
                if msg:
                    alert_types[msg.group(1)] += 1

        return {
            'total_alerts': len(alerts),
            'alert_types': dict(alert_types.most_common(10))
        }
    except FileNotFoundError:
        return {'error': 'Snort log not found'}

def analyze_suricata_logs(log_file='/var/log/suricata/eve.json'):
    """Analyze Suricata JSON logs"""
    try:
        alerts = []
        with open(log_file, 'r') as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    if entry.get('event_type') == 'alert':
                        alerts.append(entry)
                except:
                    continue

        # Categorize alerts
        alert_sigs = Counter(a.get('alert', {}).get('signature', 'Unknown') for a in alerts)

        return {
            'total_alerts': len(alerts),
            'alert_signatures': dict(alert_sigs.most_common(10))
        }
    except FileNotFoundError:
        return {'error': 'Suricata log not found'}

if __name__ == '__main__':
    print("="*70)
    print("ACEAC Log Analysis")
    print("="*70)
    print()

    print("Snort Analysis:")
    snort = analyze_snort_logs()
    print(json.dumps(snort, indent=2))
    print()

    print("Suricata Analysis:")
    suricata = analyze_suricata_logs()
    print(json.dumps(suricata, indent=2))
    print()
EOF

chmod +x ~/analyze_logs.py

echo ""
echo "========================================================================"
echo "Defense Node Setup Complete!"
echo "========================================================================"
echo ""
echo "Services installed:"
echo "  ✓ Snort IDS"
echo "  ✓ Suricata IDS"
echo "  ✓ Wazuh SIEM"
echo "  ✓ Monitoring tools"
echo ""
echo "Useful commands:"
echo "  - Monitor attacks:  ./monitor_attacks.sh"
echo "  - Analyze logs:     ./analyze_logs.py"
echo "  - View Snort logs:  sudo tail -f /var/log/snort/alert"
echo "  - View Suricata:    sudo tail -f /var/log/suricata/fast.log"
echo "  - View Wazuh:       sudo tail -f /var/ossec/logs/alerts/alerts.log"
echo ""
echo "Service status:"
sudo systemctl status snort --no-pager | grep Active
sudo systemctl status suricata --no-pager | grep Active
sudo systemctl status wazuh-manager --no-pager | grep Active
echo ""
echo "========================================================================"
