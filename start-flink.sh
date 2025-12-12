#!/bin/bash
set -e

MODE="$1"

echo "---------------------------------------------------"
echo "🚀 Starting Flink in mode: ${MODE}"
echo "Python executable: $FLINK_PYTHON_EXECUTABLE"
echo "---------------------------------------------------"

# Start JobManager
if [ "$MODE" = "jobmanager" ]; then
    echo "➡ Starting JobManager..."
    exec /opt/flink/bin/jobmanager.sh start-foreground
fi

# Start TaskManager
if [ "$MODE" = "taskmanager" ]; then
    echo "➡ Starting TaskManager..."
    exec /opt/flink/bin/taskmanager.sh start-foreground
fi

echo "❌ ERROR: You must pass either 'jobmanager' or 'taskmanager'"
exit 1
