#!/bin/bash
# Fast copy using gsutil with parallel transfers
set -e

pkill -f rsync 2>/dev/null || true
echo "Stopped rsync, switching to gsutil parallel copy..."

# Use gsutil with -m for parallel transfers
# This is MUCH faster than rsync through gcsfuse
nohup gsutil -m cp -r \
    gs://fundamental_ml_shared_storage/models/DeepSeek-V3.2-fp8/* \
    /models-local/DeepSeek-V3.2-fp8/ \
    > /tmp/copy_v32_gsutil.log 2>&1 &

echo "gsutil copy started in background. PID: $!"
echo "Monitor with: tail -f /tmp/copy_v32_gsutil.log"
echo "Check progress: ls /models-local/DeepSeek-V3.2-fp8/ | wc -l"
