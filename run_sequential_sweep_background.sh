#!/bin/bash

# Sequential Sweep Background Runner
# Runs 20 combinations sequentially across GPUs to avoid process multiplication

echo "🚀 Starting Sequential Sweep (20 combinations, sequential execution) in background..."

# Create a unique session name with timestamp
SESSION_NAME="sequential_sweep_$(date +%Y%m%d_%H%M%S)"
LOG_FILE="sequential_sweep_background_$(date +%Y%m%d_%H%M%S).log"

echo "📋 Session: $SESSION_NAME"
echo "📜 Background log: $LOG_FILE"

# Run the sweep using nohup for persistence
nohup python sequential_sweep.py > "$LOG_FILE" 2>&1 &
SWEEP_PID=$!

echo "🔄 Sequential sweep started with PID: $SWEEP_PID"
echo "📊 To monitor progress:"
echo "   tail -f $LOG_FILE"
echo ""
echo "🔍 To check if still running:"
echo "   ps aux | grep $SWEEP_PID"
echo ""
echo "🛑 To stop the sweep:"
echo "   kill $SWEEP_PID"

# Save PID for later reference
echo "$SWEEP_PID" > sequential_sweep.pid
echo "💾 PID saved to sequential_sweep.pid"

echo ""
echo "✅ Sequential sweep is now running in background!"
echo "   The process will continue even if you close this terminal."
echo "   Execution: Sequential (no process multiplication risk)"