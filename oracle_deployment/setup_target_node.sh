#!/bin/bash

################################################################################
# ACEAC Target Node Setup for Oracle Free Tier
# Sets up intentionally vulnerable services for testing
#
# Author: @sarowarzahan414
# Date: 2024-12-24
#
# WARNING: This creates vulnerable services for TESTING ONLY
# NEVER expose this to the public internet
################################################################################

set -e

echo "========================================================================"
echo "ACEAC Target Node Setup - Oracle Free Tier"
echo "========================================================================"
echo "WARNING: This will install INTENTIONALLY VULNERABLE services"
echo "This VM should ONLY be used in isolated test environments"
echo "NEVER expose to public internet"
echo "========================================================================"
echo ""
echo "Continue? (type 'yes' to confirm)"
read -r confirm
if [ "$confirm" != "yes" ]; then
    echo "Aborted"
    exit 1
fi

# Update system
echo "[1/5] Updating system..."
sudo apt update
sudo apt upgrade -y

# Install Docker
echo "[2/5] Installing Docker..."
sudo apt install -y \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg \
    lsb-release

curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/download/v2.20.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Install vulnerable services
echo "[3/5] Installing vulnerable services..."

# DVWA (Damn Vulnerable Web Application)
echo "Setting up DVWA..."
cd ~
git clone https://github.com/digininja/DVWA.git
cd DVWA

cat > docker-compose.yml <<EOF
version: '3'
services:
  dvwa:
    image: vulnerables/web-dvwa
    ports:
      - "80:80"
    environment:
      - MYSQL_DATABASE=dvwa
      - MYSQL_USER=dvwa
      - MYSQL_PASSWORD=p@ssw0rd
      - MYSQL_ROOT_PASSWORD=r00tp@ssw0rd
    restart: always
EOF

docker-compose up -d

# WebGoat (OWASP vulnerable web app)
echo "Setting up WebGoat..."
cd ~
mkdir -p webgoat
cd webgoat

cat > docker-compose.yml <<EOF
version: '3'
services:
  webgoat:
    image: webgoat/webgoat-8.0
    ports:
      - "8080:8080"
    environment:
      - WEBGOAT_PORT=8080
    restart: always
EOF

docker-compose up -d

# Mutillidae (another vulnerable web app)
echo "Setting up Mutillidae..."
cd ~
mkdir -p mutillidae
cd mutillidae

cat > docker-compose.yml <<EOF
version: '3'
services:
  mutillidae:
    image: citizenstig/nowasp
    ports:
      - "8081:80"
    restart: always
EOF

docker-compose up -d

# Basic vulnerable services
echo "[4/5] Installing basic vulnerable services..."

# FTP server (vulnerable config)
sudo apt install -y vsftpd
sudo sed -i 's/#write_enable=YES/write_enable=YES/' /etc/vsftpd.conf
sudo sed -i 's/anonymous_enable=NO/anonymous_enable=YES/' /etc/vsftpd.conf
sudo systemctl restart vsftpd

# Telnet server (inherently insecure)
sudo apt install -y telnetd
sudo systemctl enable inetd
sudo systemctl start inetd

# SSH with weak config (for testing only)
sudo apt install -y openssh-server
sudo sed -i 's/#PasswordAuthentication yes/PasswordAuthentication yes/' /etc/ssh/sshd_config
sudo sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config
sudo systemctl restart sshd

# MySQL with default credentials
sudo apt install -y mysql-server
sudo mysql <<EOF
CREATE USER 'test'@'%' IDENTIFIED BY 'test123';
GRANT ALL PRIVILEGES ON *.* TO 'test'@'%';
FLUSH PRIVILEGES;
EOF

# Create test users with weak passwords
echo "[5/5] Creating test users..."
sudo useradd -m -s /bin/bash testuser1
echo "testuser1:password123" | sudo chpasswd

sudo useradd -m -s /bin/bash admin
echo "admin:admin" | sudo chpasswd

# Configure firewall (Oracle Cloud specific)
echo "Configuring firewall rules..."
sudo iptables -I INPUT -p tcp --dport 80 -j ACCEPT
sudo iptables -I INPUT -p tcp --dport 8080 -j ACCEPT
sudo iptables -I INPUT -p tcp --dport 8081 -j ACCEPT
sudo iptables -I INPUT -p tcp --dport 21 -j ACCEPT
sudo iptables -I INPUT -p tcp --dport 23 -j ACCEPT
sudo iptables -I INPUT -p tcp --dport 22 -j ACCEPT
sudo iptables -I INPUT -p tcp --dport 3306 -j ACCEPT

# Save iptables rules
sudo netfilter-persistent save

echo ""
echo "========================================================================"
echo "Target Node Setup Complete!"
echo "========================================================================"
echo ""
echo "Vulnerable services running:"
echo "  - DVWA:       http://$(hostname -I | awk '{print $1}'):80"
echo "  - WebGoat:    http://$(hostname -I | awk '{print $1}'):8080"
echo "  - Mutillidae: http://$(hostname -I | awk '{print $1}'):8081"
echo "  - FTP:        ftp://$(hostname -I | awk '{print $1}'):21"
echo "  - Telnet:     telnet://$(hostname -I | awk '{print $1}'):23"
echo "  - SSH:        ssh://$(hostname -I | awk '{print $1}'):22"
echo "  - MySQL:      $(hostname -I | awk '{print $1}'):3306"
echo ""
echo "Test credentials:"
echo "  - testuser1:password123"
echo "  - admin:admin"
echo "  - MySQL: test:test123"
echo ""
echo "WARNING: These services are INTENTIONALLY VULNERABLE"
echo "DO NOT expose this VM to the internet!"
echo "========================================================================"
