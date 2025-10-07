#!/bin/bash

# Simple Sweep Background Runner
# This script runs the simple sweep in the background and persists across terminal sessions

echo "🚀 Starting Simple Sweep (20 combinations, 4 per GPU) in background..."

# Create a unique session name with timestamp
SESSION_NAME="simple_sweep_$(date +%Y%m%d_%H%M%S)"
LOG_FILE="simple_sweep_background_$(date +%Y%m%d_%H%M%S).log"

echo "📋 Session: $SESSION_NAME"
echo "📜 Background log: $LOG_FILE"

# Run the sweep using nohup for persistence
nohup python simple_sweep.py > "$LOG_FILE" 2>&1 &
SWEEP_PID=$!

echo "🔄 Simple sweep started with PID: $SWEEP_PID"
echo "📊 To monitor progress:"
echo "   tail -f $LOG_FILE"
echo ""
echo "🔍 To check if still running:"
echo "   ps aux | grep $SWEEP_PID"
echo ""
echo "🛑 To stop the sweep:"
echo "   kill $SWEEP_PID"

# Save PID for later reference
echo "$SWEEP_PID" > simple_sweep.pid
echo "💾 PID saved to simple_sweep.pid"

echo ""
echo "✅ Simple sweep is now running in background!"
echo "   The process will continue even if you close this terminal."