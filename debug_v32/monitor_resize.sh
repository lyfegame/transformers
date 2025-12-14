#!/bin/bash
# Monitor H200 MIG resize request
# Usage: ./monitor_resize.sh [REQUEST_NAME]

MIG="h200-mig-cluster"
ZONE="europe-west1-b"
PROJECT="fundamental-labs"

# Get latest resize request if not specified
if [ -z "$1" ]; then
  REQUEST=$(gcloud compute instance-groups managed resize-requests list \
    --project=$PROJECT --filter="instanceGroupManager:$MIG" \
    --format="value(name)" 2>/dev/null | head -1)
  if [ -z "$REQUEST" ]; then
    echo "No resize requests found. Checking instance status only."
    gcloud compute instances list --filter="name~$MIG" --project=$PROJECT \
      --format="table(name,status,networkInterfaces[0].networkIP)"
    exit 0
  fi
else
  REQUEST="$1"
fi

echo "Monitoring resize request: $REQUEST"
echo ""

for i in $(seq 1 40); do
  echo "--- Check $i at $(date +%H:%M:%S) ---"

  state=$(gcloud compute instance-groups managed resize-requests describe $MIG \
    --resize-request=$REQUEST --zone=$ZONE --project=$PROJECT \
    --format="value(state)" 2>/dev/null)

  echo "State: $state"

  gcloud compute instances list --filter="name~$MIG" --project=$PROJECT \
    --format="table(name,status,networkInterfaces[0].networkIP)" 2>/dev/null

  if [ "$state" = "SUCCEEDED" ]; then
    echo ""
    echo "SUCCESS! All nodes provisioned."
    break
  fi

  echo ""
  sleep 30
done
