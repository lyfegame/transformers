#!/bin/bash
# Setup SSH keys between nodes

PUBKEY="ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABgQC6EiDDHg7Rw6zo3PUlXp6tPNWg6Vo/sOgrKJ6vhFxjzWJDcQiqIpaA3PcTR91ORy6+HLtnGbxYDd1VB8Ahehlal7t1JXC2VG6Srwdn0muNysJjyTW0MxfvEOHGXSaANp0q5EoW2LcxEFt/4PKb6+T3EL4n0KIz/5lBvH8GCh0EB4Pb4qwjT3qETTVeVpUyAkCc9C2/KUgXTiH/w7PNWjsbix75Jz6w9SvI/LLttKyuZg+fZ1fKBY5AhGF15IC8x8CmSz0YNSNmxzygoXEMDCzS2msQGaGRiAf0wgpfgotr3l2RK6Cd4DPce4VvJMRReICScxiMFubpS3fpx2/QOGazD4j7rq64NfayCui1X//+dP1MEvQJssTLj9ihrUNTBuu3UGPx/x+Ch5UNNTRrzoxMK1Gh42ocJHjo/482YwP7miOAioZ47PUJHGnYrVkf/v2V4aw12V7Qc2h8Fevqf82IBzJTG9Ngcsm9yHrauLhySevqmy2az9DGSy4P1he9k18= shuyingluo@h200-mig-cluster-m019"

NODES="h200-mig-cluster-m019 h200-mig-cluster-8slj h200-mig-cluster-lgsn h200-mig-cluster-fm3h h200-mig-cluster-rn1h"
ZONE="europe-west1-b"
PROJECT="fundamental-labs"

for node in $NODES; do
  echo "Adding key to $node..."
  gcloud compute ssh $node --zone=$ZONE --project=$PROJECT -- "mkdir -p ~/.ssh && echo '$PUBKEY' >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys" 2>&1 | grep -v "Pseudo-terminal\|known_hosts"
done

echo "Done adding keys to all nodes"
