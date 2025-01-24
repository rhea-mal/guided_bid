#!/bin/bash

# Define the partition name
# PARTITION_NAME="iris"
PARTITION_NAME="iris-hi"

# Get all job IDs from the specified partition
JOB_IDS=$(squeue --noheader --format="%A %P" | awk -v partition="$PARTITION_NAME" '$2 == partition {print $1}')

# Cancel each job
for JOB_ID in $JOB_IDS; do
    scancel $JOB_ID
done

echo "All jobs on partition $PARTITION_NAME have been cancelled."
