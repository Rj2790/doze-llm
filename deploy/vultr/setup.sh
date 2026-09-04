#!/usr/bin/env bash
# One-time VM setup (Ubuntu 22.04, Vultr GPU plan). Run as root after first login.
# Installs the NVIDIA driver if missing, Python 3.11 venv, CUDA torch, and the repo deps.
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get install -y python3.11 python3.11-venv python3-pip git rsync tmux htop
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "installing NVIDIA driver (reboot required afterwards)"
  apt-get install -y ubuntu-drivers-common
  ubuntu-drivers install --gpgpu || apt-get install -y nvidia-driver-535-server
  echo "REBOOT_REQUIRED"
fi
mkdir -p /data/results /data/hf-cache /opt/doze
cd /opt/doze
python3.11 -m venv .venv
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q torch --index-url https://download.pytorch.org/whl/cu121
.venv/bin/pip install -q "transformers>=4.56" "peft>=0.15" accelerate safetensors pytest
echo 'export HF_HOME=/data/hf-cache' >> /root/.bashrc
echo 'export DOZE_RESULTS=/data/results' >> /root/.bashrc
echo 'export TOKENIZERS_PARALLELISM=false' >> /root/.bashrc
echo "SETUP_DONE"
