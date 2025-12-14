# H200 Cluster Quick Reference

## Current Cluster Status (MIG: h200-mig-cluster)

> **Last Updated: 2025-12-13**

| Node | Name | Internal IP | External IP | node_rank |
|------|------|-------------|-------------|-----------|
| Master | h200-mig-cluster-m019 | 10.0.0.13 | 34.34.140.147 | 0 |
| Worker 1 | h200-mig-cluster-8slj | 10.0.0.14 | 35.205.102.22 | 1 |
| Worker 2 | h200-mig-cluster-lgsn | 10.0.0.15 | 35.233.90.41 | 2 |
| Worker 3 | h200-mig-cluster-fm3h | 10.0.0.16 | 35.187.119.120 | 3 |
| Worker 4 | h200-mig-cluster-rn1h | 10.0.0.17 | 34.52.158.56 | 4 |

**Total: 40 GPUs (8 H200s × 5 nodes)**

### GCloud Config
```bash
PROJECT_ID="fundamental-labs"
ZONE="europe-west1-b"
MIG_NAME="h200-mig-cluster"
```

---

## SSH Access

```bash
# SSH to master node
gcloud compute ssh h200-mig-cluster-m019 --zone=europe-west1-b --project=fundamental-labs

# SSH to any node by name
gcloud compute ssh h200-mig-cluster-8slj --zone=europe-west1-b --project=fundamental-labs
```

---

## Starting/Stopping Cluster

```bash
# Start all nodes
gcloud compute instances start h200-mig-cluster-m019 h200-mig-cluster-8slj h200-mig-cluster-lgsn h200-mig-cluster-fm3h h200-mig-cluster-rn1h --zone=europe-west1-b --project=fundamental-labs

# Stop all nodes (to save costs)
gcloud compute instances stop h200-mig-cluster-m019 h200-mig-cluster-8slj h200-mig-cluster-lgsn h200-mig-cluster-fm3h h200-mig-cluster-rn1h --zone=europe-west1-b --project=fundamental-labs

# Check status
gcloud compute instances list --filter="name~h200-mig-cluster" --project=fundamental-labs --format="table(name,zone,status)"
```

---

## The Golden Rule: Fire-and-Forget

**NEVER rely on long-running SSH sessions. NEVER stream logs over SSH.**

The internet is unstable. SSH connections drop. Instead:
1. Copy script to remote machine
2. Execute script in background (nohup + &)
3. All output goes to timestamped log files
4. Check logs with separate, short SSH commands (tail, not streaming)

```bash
# BAD: Streaming logs, will break when connection drops
gcloud compute ssh master -- "tail -f /tmp/training.log"

# GOOD: Get a snapshot of logs
gcloud compute ssh h200-mig-cluster-m019 --zone=europe-west1-b -- "tail -100 /tmp/v3_training_*/node0.log"
```

---

## Essential Commands (All Non-Blocking)

```bash
# Kill training on all nodes
gcloud compute ssh h200-mig-cluster-m019 --zone=europe-west1-b -- \
    "for ip in 10.0.0.13 10.0.0.14 10.0.0.15 10.0.0.16 10.0.0.17; do ssh -o StrictHostKeyChecking=no \$ip 'pkill -9 -f torchrun' & done; wait; echo done"

# Check process count per node (snapshot, not streaming)
gcloud compute ssh h200-mig-cluster-m019 --zone=europe-west1-b -- \
    "for ip in 10.0.0.13 10.0.0.14 10.0.0.15 10.0.0.16 10.0.0.17; do echo -n \"\$ip: \"; ssh -o ConnectTimeout=5 \$ip 'pgrep -c torchrun 2>/dev/null || echo 0'; done"

# Get last 50 lines of log (snapshot)
gcloud compute ssh h200-mig-cluster-m019 --zone=europe-west1-b -- \
    "tail -50 /tmp/v3_training_*/node0.log"

# List available log directories
gcloud compute ssh h200-mig-cluster-m019 --zone=europe-west1-b -- \
    "ls -la /tmp/v3_training_*"

# Check ALL nodes for errors
gcloud compute ssh h200-mig-cluster-m019 --zone=europe-west1-b -- \
    "for ip in 10.0.0.13 10.0.0.14 10.0.0.15 10.0.0.16 10.0.0.17; do echo '=== '\$ip' ==='; ssh -o ConnectTimeout=5 \$ip 'tail -10 /tmp/v3_training_*/node*.log 2>/dev/null | tail -5'; done"

# Check dmesg for OOM on ALL nodes
gcloud compute ssh h200-mig-cluster-m019 --zone=europe-west1-b -- \
    "for ip in 10.0.0.13 10.0.0.14 10.0.0.15 10.0.0.16 10.0.0.17; do echo '=== '\$ip' ==='; ssh \$ip 'dmesg | grep -i oom | tail -3'; done"
```

---

## Verify Setup (After Starting Cluster)

```bash
# SSH to master
gcloud compute ssh h200-mig-cluster-m019 --zone=europe-west1-b --project=fundamental-labs

# On the node:
source /etc/profile.d/nccl.sh   # ALWAYS do this first
nvidia-smi -L                    # Check 8 GPUs visible
ibv_devices                      # Check RDMA devices
ls /models                       # Check GCS mount
python3 -c "import torch; print(f'PyTorch {torch.__version__}, GPUs: {torch.cuda.device_count()}')"
```

---

## Critical NCCL Settings

These are auto-configured in `/etc/profile.d/nccl.sh` but critical to understand:

```bash
# MUST use GID index 3 for GCP RoCEv2
export NCCL_IB_GID_INDEX=3

# Enable GPUDirect RDMA
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=PIX
export NCCL_NET_GDR_READ=1

# Reliability (prevents silent hangs)
export NCCL_IB_TIMEOUT=22
export NCCL_IB_RETRY_CNT=7

# Socket interface for OOB communication (10.0.0.x network)
export NCCL_SOCKET_IFNAME=enp0s19   # CRITICAL: Must match interface with 10.0.0.x IP
```

---

## Launching Multi-Node Training

### Key Rules
1. **Always source NCCL env**: `source /etc/profile.d/nccl.sh`
2. **Start master FIRST**: Workers connect to master's rendezvous endpoint
3. **Use timestamped log directories**: `/tmp/v3_training_$(date +%Y%m%d_%H%M%S)/`
4. **Use nohup + &**: Don't rely on SSH session staying alive

### Example: 5-Node Training Launch

```bash
# On master node (10.0.0.13):
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="/tmp/v3_training_${TIMESTAMP}"
MASTER_ADDR="10.0.0.13"
MASTER_PORT="29500"
SCRIPT="~/training/train_script.py"

# Create log dirs on all nodes
mkdir -p "$LOG_DIR"
for ip in 10.0.0.14 10.0.0.15 10.0.0.16 10.0.0.17; do
    ssh -o StrictHostKeyChecking=no $ip "mkdir -p $LOG_DIR" &
done
wait

# Start MASTER FIRST
source /etc/profile.d/nccl.sh
nohup torchrun --nproc_per_node=8 --nnodes=5 --node_rank=0 \
    --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
    --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    $SCRIPT > "$LOG_DIR/node0.log" 2>&1 &

echo "Master started, waiting 5s..."
sleep 5

# Start workers (background SSH)
ssh 10.0.0.14 "source /etc/profile.d/nccl.sh && nohup torchrun --nproc_per_node=8 --nnodes=5 --node_rank=1 --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT $SCRIPT > $LOG_DIR/node1.log 2>&1 &" &
ssh 10.0.0.15 "source /etc/profile.d/nccl.sh && nohup torchrun --nproc_per_node=8 --nnodes=5 --node_rank=2 --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT $SCRIPT > $LOG_DIR/node2.log 2>&1 &" &
ssh 10.0.0.16 "source /etc/profile.d/nccl.sh && nohup torchrun --nproc_per_node=8 --nnodes=5 --node_rank=3 --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT $SCRIPT > $LOG_DIR/node3.log 2>&1 &" &
ssh 10.0.0.17 "source /etc/profile.d/nccl.sh && nohup torchrun --nproc_per_node=8 --nnodes=5 --node_rank=4 --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT $SCRIPT > $LOG_DIR/node4.log 2>&1 &" &
wait

echo "All nodes started. Logs in: $LOG_DIR"
```

---

## Storage

| Path | Type | Speed | Contents |
|------|------|-------|----------|
| `/models-local` | NFS (20TB hyperdisk) | **FAST** | Local copies of models (prefer this!) |
| `/models` | GCS (gcsfuse) | Slow | All models from GCS bucket |
| `/data/checkpoints` | Local SSD | Fast | Training outputs |

### Recommended: Use `/models-local` (Fast Local Disk)
The 20TB `models-disk` is attached to master and shared via NFS to all nodes:
```
/models-local/
├── DeepSeek-V2-Lite/
├── DeepSeek-V3-bf16/
├── DeepSeek-V3.2-fp8/       # (copying in progress)
└── deepseek-ai--DeepSeek-V2/
```

**Usage in training:**
```python
model_path = "/models-local/DeepSeek-V3.2-fp8"
model = AutoModelForCausalLM.from_pretrained(model_path)
```

### Storage Architecture
```
Master Node (10.0.0.13):
  /dev/nvme33n1 (20TB) → /mnt/models-disk → /models-local (symlink)
                              ↓
                         NFS Export
                              ↓
Worker Nodes (10.0.0.14-17):
  NFS mount 10.0.0.13:/mnt/models-disk → /mnt/models-disk → /models-local (symlink)
```

### Fallback: GCS (gcsfuse) - Slower
If `/models` shows permission denied, remount with:
```bash
sudo umount /mnt/gcs/models 2>/dev/null
sudo gcsfuse -o allow_other --implicit-dirs --only-dir models fundamental_ml_shared_storage /mnt/gcs/models
```

GCS models (slower, but has everything):
```
/models/
├── DeepSeek-V3-bf16/
├── DeepSeek-V3.2-fp8/
├── deepseek-ai--DeepSeek-V2/
├── deepseek-ai--DeepSeek-V2-Lite/
└── deprecated/
```

---

## Torchrun Reference

```bash
torchrun \
    --nproc_per_node=8 \              # GPUs per node (8 for H200)
    --nnodes=5 \                       # Total nodes
    --node_rank=0 \                    # This node's rank (0=master)
    --master_addr=10.0.0.13 \          # Master IP
    --master_port=29500 \              # Master port
    --rdzv_backend=c10d \              # Rendezvous backend
    --rdzv_endpoint=10.0.0.13:29500 \  # Same as master addr:port
    --rdzv-conf timeout=7200 \         # 2hr timeout for workers to join
    train_script.py
```

---

## Anti-Patterns to Avoid

| Don't | Do Instead |
|-------|------------|
| `tail -f` over SSH | `tail -100` snapshot |
| Stream logs over SSH | Use WandB for real-time, snapshots for details |
| Run `watch` commands over SSH | Single snapshot commands |
| Complex inline SSH commands | Copy script, execute, check log |
| Guess which node failed | Check ALL nodes with facts |
| Use non-timestamped log paths | Always `$(date +%Y%m%d_%H%M%S)` |
| Start workers before master | Master first, wait 5s, then workers |

---

## Troubleshooting

### NCCL Hangs
1. Verify GID index: `cat /sys/class/infiniband/rocep*/ports/1/gids/3`
2. Check NCCL debug: `export NCCL_DEBUG=INFO`
3. Test RDMA: `ibv_rc_pingpong -d rocep145s0 -g 3` (between nodes)

### GCS Mount Issues
```bash
# Check if mounted
mountpoint /mnt/gcs/models

# Manual mount
gcsfuse --implicit-dirs --only-dir models fundamental_ml_shared_storage /mnt/gcs/models

# Check bucket access
gsutil ls gs://fundamental_ml_shared_storage/models/
```

### RendezvousConnectionError
Workers can't connect to master. Solutions:
- Ensure master started first and waited 5s
- Check master is listening: `ss -tlnp | grep 29500`
- Verify network connectivity between nodes

### OOM Errors
Check all nodes for OOM:
```bash
gcloud compute ssh h200-mig-cluster-m019 --zone=europe-west1-b -- \
    "for ip in 10.0.0.13 10.0.0.14 10.0.0.15 10.0.0.16 10.0.0.17; do echo '=== '\$ip' ==='; ssh \$ip 'dmesg | grep -i oom | tail -3'; done"
```
